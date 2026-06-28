// Generate an EE-SPACE figure-8 tracking problem for the corrected iiwa14 + gravity, mirroring GATO's
// run_mpc_fig8: the reference is a smooth Cartesian figure-8 of the END-EFFECTOR POSITION (it does NOT
// command any joint motion — critical, see the nullspace note in main), and the warm-start is a
// gravity-compensated HOLD at q_start. grid.cuh's own FK gives the EE center and ID gives the hold
// control, so everything is consistent with the model the solver and sim use. Writes MPCGPU trajfiles:
//   <prefix>_eepos.traj : per-row 6-wide EE pose (figure-8 xyz + held orientation; cost uses xyz)
//   <prefix>_traj.csv   : per-row 21-wide [q(7) qd(7) u(7)]  (constant hold warm-start)
// Args: <prefix> <amp_scale: 0=>regulation/hold, 1=>full figure-8> <period_s>.
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include "dynamics/rbd_plant.cuh"
#include "settings.cuh"   // TIMESTEP — keep the reference spacing locked to the tracker's integration dt

template<typename T>
__global__ void k_id(T* d_u, const T* d_q, const T* d_qd, const T* d_qdd, void* d_rm, T gravity){
    __shared__ T s_q[7], s_qd[7], s_qdd[7], s_u[7];
    for(int i=threadIdx.x;i<7;i+=blockDim.x){ s_q[i]=d_q[i]; s_qd[i]=d_qd[i]; s_qdd[i]=d_qdd[i]; }
    __syncthreads();
    grid::inverse_dynamics_device<T>(s_u, s_q, s_qd, s_qdd, (grid::robotModel<T>*)d_rm, /*d_f_ext*/nullptr, gravity);
    __syncthreads();
    for(int i=threadIdx.x;i<7;i+=blockDim.x) d_u[i]=s_u[i];
}
template<typename T>
__global__ void k_fk(T* d_ee, const T* d_q, void* d_rm){
    __shared__ T s_q[7], s_ee[6];
    for(int i=threadIdx.x;i<7;i+=blockDim.x){ s_q[i]=d_q[i]; }
    __syncthreads();
    grid::end_effector_pose_device<T>(s_ee, s_q, (grid::robotModel<T>*)d_rm);
    __syncthreads();
    for(int i=threadIdx.x;i<6;i+=blockDim.x) d_ee[i]=s_ee[i];
}

int main(int argc, char** argv){
    using T = float;
    const int NSTEPS = 700;
    const double dt = TIMESTEP;   // shared with the tracker via settings.cuh
    const T gravity = gato_plant::GRAVITY<T>();
    std::string prefix = (argc > 1) ? argv[1] : "examples/trajfiles/0_0";

    // ---------------------------------------------------------------------------------------------
    // EE-SPACE figure-8 reference + constant "hold" warm-start (mirrors GATO's run_mpc_fig8 setup).
    // RATIONALE: the iiwa is a 7-DOF arm tracking a 3-DOF EE-POSITION task, so there is a 4-D cost
    // nullspace — including joint 7, whose EE-position Jacobian column is ~0 and whose position is
    // NOT penalized (s_Q[q-block]=0). A reference that COMMANDS joint motion (esp. on joint 7, the
    // stiff Minv[6,6]≈392 mode) is untrackable by an EE-only cost → the nullspace runs away → the
    // closed loop diverges (verified: even the exact QDLDL solve diverges on a joint-space sinusoid).
    // So the reference must live in EE space and NEVER command the nullspace joints; the solver then
    // discovers a joint trajectory and the nullspace stays bounded (qd_cost damps it). The warm-start
    // is a gravity-compensated hold at q_start — the solver bends it toward the moving EE figure-8.
    double q0[7] = {0.40, 0.80, 0.30, -1.10, 0.50, 0.60, 0.30};   // start config (within iiwa14 limits)
    // EE figure-8 (GATO figure8): dx = A*sin(wt), dz = A*sin(2wt)/2, rotated theta about z, added to
    // the q_start EE position so it stays in the reachable workspace. amp_scale 0 => pure hold/regulation.
    double A_default = 0.12;              // figure-8 amplitude (m); modest => reachable + gently trackable
    double period    = 4.0;              // seconds per figure-8 cycle
    double theta     = M_PI/4.0;          // rotation of the figure-8 plane about z (GATO default)
    double amp_scale = (argc > 2) ? atof(argv[2]) : 1.0;
    if (argc > 3) period = atof(argv[3]);
    double A = A_default * amp_scale;
    double omega = 2.0 * M_PI / period;

    auto* d_rm = grid::init_robotModel<T>();
    T *d_q,*d_qd,*d_qdd,*d_u,*d_ee;
    cudaMalloc(&d_q,7*sizeof(T)); cudaMalloc(&d_qd,7*sizeof(T)); cudaMalloc(&d_qdd,7*sizeof(T));
    cudaMalloc(&d_u,7*sizeof(T)); cudaMalloc(&d_ee,6*sizeof(T));

    // --- compute the EE center (FK at q_start) and the hold control (ID at q_start, qd=qdd=0) ONCE ---
    T hq0[7], hqz[7] = {0,0,0,0,0,0,0};
    for(int j=0;j<7;j++) hq0[j] = (T)q0[j];
    size_t id_smem = grid::INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_COUNT*sizeof(T);
    size_t fk_smem = grid::END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_COUNT*sizeof(T);
    cudaMemcpy(d_q,hq0,7*sizeof(T),cudaMemcpyHostToDevice);
    cudaMemcpy(d_qd,hqz,7*sizeof(T),cudaMemcpyHostToDevice);
    cudaMemcpy(d_qdd,hqz,7*sizeof(T),cudaMemcpyHostToDevice);
    k_fk<T><<<1,32,fk_smem>>>(d_ee,d_q,d_rm);
    k_id<T><<<1,32,id_smem>>>(d_u,d_q,d_qd,d_qdd,d_rm,gravity);
    cudaDeviceSynchronize();
    T center[6], u_hold[7];
    cudaMemcpy(center,d_ee,6*sizeof(T),cudaMemcpyDeviceToHost);
    cudaMemcpy(u_hold,d_u,7*sizeof(T),cudaMemcpyDeviceToHost);

    FILE* fee = fopen((prefix+"_eepos.traj").c_str(), "w");
    FILE* fxu = fopen((prefix+"_traj.csv").c_str(), "w");
    if(!fee || !fxu){ printf("cannot open output files\n"); return 1; }

    for(int t=0; t<NSTEPS; t++){
        double tm = t*dt;
        // figure-8 delta in the (x,z) plane, rotated theta about z, added to the held EE center
        double dx_u = A*sin(omega*tm);
        double dz   = A*sin(2.0*omega*tm)/2.0;
        double dx   =  cos(theta)*dx_u;   // rotate [dx_u, 0, dz] about z (z-component unchanged)
        double dy   =  sin(theta)*dx_u;
        T hee[6];
        hee[0] = (T)(center[0] + dx);
        hee[1] = (T)(center[1] + dy);
        hee[2] = (T)(center[2] + dz);
        hee[3] = center[3]; hee[4] = center[4]; hee[5] = center[5];   // hold orientation (cost uses xyz)
        // eepos.traj row (6-wide EE pose; cost reads xyz)
        for(int i=0;i<6;i++) fprintf(fee, "%.9g%s", hee[i], i<5?",":"\n");
        // xu_traj row: constant hold warm-start [q_start, 0, gravity-comp(q_start)]
        for(int j=0;j<7;j++) fprintf(fxu, "%.9g,", q0[j]);
        for(int j=0;j<7;j++) fprintf(fxu, "%.9g,", 0.0);
        for(int j=0;j<7;j++) fprintf(fxu, "%.9g%s", u_hold[j], j<6?",":"\n");
    }
    fclose(fee); fclose(fxu);
    printf("wrote %s_eepos.traj + %s_traj.csv (%d steps, EE fig-8 A=%.3f m, period=%.2f s, gravity=%.3f)\n",
           prefix.c_str(), prefix.c_str(), NSTEPS, A, period, gravity);
    printf("  EE center (FK q_start) = [%.4f %.4f %.4f]\n", center[0], center[1], center[2]);
    return 0;
}
