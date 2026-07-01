// Generate the SHARED iiwa14 EE-space figure-8 tracking reference for the fair 3-way benchmark
// (GATO / MPCGPU / BatchThneed all track the IDENTICAL fig8). Authored directly in EE space so it
// never commands joint motion (an EE-position-only cost has a 4-D nullspace incl. the stiff joint 7;
// a joint-commanding reference excites it and diverges — see docs/modernization.md). The MPC discovers
// the joints that realize the EE path. Mirrors GATO's run_mpc_fig8:
//   - goal:       ee(t) = center + [A*sin(wt), 0, 0.5*A*sin(2wt)]   (theta=0 vertical fig8, y=const)
//                 center = FK(q0); so ee(0)=center => ZERO initial tracking error.
//   - warm-start: x_curr replicated with ZERO controls: every knot = [q0, 0, 0]. This EXACTLY mirrors
//                 GATO's run_mpc_fig8 (XU state=x_curr, controls=0) and the CPU baseline. It is
//                 dynamically INFEASIBLE (zero torque vs gravity) ON PURPOSE: a gravity-comp feasible
//                 hold is a strict local MINIMUM of the SQP merit (any step raises the integrator
//                 defect -> line search rejects it -> the solver is trapped and never tracks). The
//                 infeasible start gives the line search a large defect to reduce, so the first step
//                 is a descent direction toward feasibility+tracking (this is why GATO/CPU track).
// Writes MPCGPU trajfiles (consumed by track_iiwa_pcg.cu / validate_track.cu):
//   <prefix>_eepos.traj : per-row 6-wide EE pose [x y z 0 0 0] (cost reads xyz)
//   <prefix>_traj.csv   : per-row 21-wide [q(7) qd(7) u(7)] = static hold (d_xs = first row's [q,qd])
// Args: <prefix> <A: EE amplitude m, default 0.15> <period_s: default 6.0>.
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
    const double dt = TIMESTEP;   // shared with the tracker via settings.cuh
    const T gravity = gato_plant::GRAVITY<T>();
    std::string prefix = (argc > 1) ? argv[1] : "examples/trajfiles/0_0";
    double A      = (argc > 2) ? atof(argv[2]) : 0.15;   // EE fig8 amplitude (m)
    double period = (argc > 3) ? atof(argv[3]) : 6.0;    // seconds per fig8 cycle
    double omega  = 2.0 * M_PI / period;
    const int NSTEPS = (int)(2.0 * period / dt) + 2;      // ~2 fig8 cycles of reference

    // start config "readyC" — bent, EE ~ [0.51, 0, 0.51] (mid forward workspace; matches GATO tune).
    double q0d[7] = {0.0, 0.30, 0.0, -1.60, 0.0, 1.20, 0.0};
    T q0[7]; for(int j=0;j<7;j++) q0[j]=(T)q0d[j];

    auto* d_rm = grid::init_robotModel<T>();
    T *d_q,*d_qd,*d_qdd,*d_u,*d_ee;
    cudaMalloc(&d_q,7*sizeof(T)); cudaMalloc(&d_qd,7*sizeof(T)); cudaMalloc(&d_qdd,7*sizeof(T));
    cudaMalloc(&d_u,7*sizeof(T)); cudaMalloc(&d_ee,6*sizeof(T));
    size_t id_smem = grid::INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_COUNT*sizeof(T);
    size_t fk_smem = grid::END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_COUNT*sizeof(T);

    // center = FK(q0); u_grav = ID(q0, 0, 0) (gravity-comp hold torque)
    T zero7[7] = {0,0,0,0,0,0,0};
    cudaMemcpy(d_q,  q0,   7*sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qd, zero7,7*sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qdd,zero7,7*sizeof(T), cudaMemcpyHostToDevice);
    k_fk<T><<<1,32,fk_smem>>>(d_ee,d_q,d_rm);
    k_id<T><<<1,32,id_smem>>>(d_u,d_q,d_qd,d_qdd,d_rm,gravity);
    cudaDeviceSynchronize();
    T center[6], u_grav[7];
    cudaMemcpy(center, d_ee, 6*sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(u_grav, d_u,  7*sizeof(T), cudaMemcpyDeviceToHost);

    FILE* fee = fopen((prefix+"_eepos.traj").c_str(), "w");
    FILE* fxu = fopen((prefix+"_traj.csv").c_str(), "w");
    if(!fee || !fxu){ printf("cannot open output files\n"); return 1; }

    for(int t=0; t<NSTEPS; t++){
        double wt = omega * (t*dt);
        double ee_x = center[0] + A*sin(wt);
        double ee_y = center[1];
        double ee_z = center[2] + 0.5*A*sin(2.0*wt);
        // goal row (6-wide EE pose; cost reads xyz, orientation target zeroed like GATO)
        fprintf(fee, "%.9g,%.9g,%.9g,0,0,0\n", ee_x, ee_y, ee_z);
        // warm-start row: x_curr replicated with ZERO controls [q0, 0, 0] — mirrors GATO/CPU (see
        // header). Infeasible on purpose so the SQP has a defect to reduce and the first step is a
        // descent direction (a gravity-comp hold is a strict merit min and traps the solver).
        for(int j=0;j<7;j++) fprintf(fxu, "%.9g,", (double)q0[j]);
        for(int j=0;j<7;j++) fprintf(fxu, "0,");
        for(int j=0;j<7;j++) fprintf(fxu, "%s", j<6?"0,":"0\n");
    }
    fclose(fee); fclose(fxu);
    printf("wrote %s_eepos.traj + %s_traj.csv (%d steps, EE fig8 A=%.3f period=%.2fs, gravity=%.3f)\n",
           prefix.c_str(), prefix.c_str(), NSTEPS, A, period, gravity);
    printf("  center=FK(q0)=[%.4f %.4f %.4f]  ee(0)=[%.4f %.4f %.4f]  u_grav=[%.3f %.3f %.3f %.3f %.3f %.3f %.3f]\n",
           center[0],center[1],center[2], center[0]+A*sin(0.0),center[1],center[2],
           u_grav[0],u_grav[1],u_grav[2],u_grav[3],u_grav[4],u_grav[5],u_grav[6]);
    return 0;
}
