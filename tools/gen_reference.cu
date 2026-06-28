// Generate a SELF-CONSISTENT reference trajectory for the corrected iiwa14 + gravity, using
// grid.cuh's own inverse dynamics (controls) and forward kinematics (EE pose) — so the reference
// is an exact, dynamically-feasible trajectory of the same model the solver and sim use. A smooth
// per-joint sinusoid keeps it inside joint limits. Writes MPCGPU trajfiles:
//   <prefix>_eepos.traj : per-row 6-wide EE pose (xyz + orientation; cost uses xyz)
//   <prefix>_traj.csv   : per-row 21-wide [q(7) qd(7) u(7)]  (state + gravity-consistent control)
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include "dynamics/rbd_plant.cuh"

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
    const double dt = 0.015625;
    const T gravity = gato_plant::GRAVITY<T>();
    std::string prefix = (argc > 1) ? argv[1] : "examples/trajfiles/0_0";

    // start config (within iiwa14 joint limits) + smooth per-joint sinusoid
    double q0[7]   = {0.40, 0.80, 0.30, -1.10, 0.50, 0.60, 0.30};
    double amp[7]  = {0.35, 0.30, 0.35, 0.30, 0.40, 0.45, 0.50};
    double period  = 4.0;                // seconds
    // argv[2] = amplitude scale (0 => regulation/hold); argv[3] = period seconds.
    double amp_scale = (argc > 2) ? atof(argv[2]) : 1.0;
    if (argc > 3) period = atof(argv[3]);
    for(int j=0;j<7;j++) amp[j] *= amp_scale;
    double omega = 2.0 * M_PI / period;

    auto* d_rm = grid::init_robotModel<T>();
    T *d_q,*d_qd,*d_qdd,*d_u,*d_ee;
    cudaMalloc(&d_q,7*sizeof(T)); cudaMalloc(&d_qd,7*sizeof(T)); cudaMalloc(&d_qdd,7*sizeof(T));
    cudaMalloc(&d_u,7*sizeof(T)); cudaMalloc(&d_ee,6*sizeof(T));

    FILE* fee = fopen((prefix+"_eepos.traj").c_str(), "w");
    FILE* fxu = fopen((prefix+"_traj.csv").c_str(), "w");
    if(!fee || !fxu){ printf("cannot open output files\n"); return 1; }

    for(int t=0; t<NSTEPS; t++){
        double tm = t*dt;
        T hq[7],hqd[7],hqdd[7];
        for(int j=0;j<7;j++){
            hq[j]   = (T)(q0[j] + amp[j]*sin(omega*tm + j*M_PI/4.0));
            hqd[j]  = (T)(amp[j]*omega*cos(omega*tm + j*M_PI/4.0));
            hqdd[j] = (T)(-amp[j]*omega*omega*sin(omega*tm + j*M_PI/4.0));
        }
        cudaMemcpy(d_q,hq,7*sizeof(T),cudaMemcpyHostToDevice);
        cudaMemcpy(d_qd,hqd,7*sizeof(T),cudaMemcpyHostToDevice);
        cudaMemcpy(d_qdd,hqdd,7*sizeof(T),cudaMemcpyHostToDevice);
        size_t id_smem = grid::INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_COUNT*sizeof(T);
        size_t fk_smem = grid::END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_COUNT*sizeof(T);
        k_id<T><<<1,32,id_smem>>>(d_u,d_q,d_qd,d_qdd,d_rm,gravity);
        k_fk<T><<<1,32,fk_smem>>>(d_ee,d_q,d_rm);
        cudaDeviceSynchronize();
        T hu[7],hee[6];
        cudaMemcpy(hu,d_u,7*sizeof(T),cudaMemcpyDeviceToHost);
        cudaMemcpy(hee,d_ee,6*sizeof(T),cudaMemcpyDeviceToHost);
        // eepos.traj row
        for(int i=0;i<6;i++) fprintf(fee, "%.9g%s", hee[i], i<5?",":"\n");
        // xu_traj row: q, qd, u
        for(int j=0;j<7;j++) fprintf(fxu, "%.9g,", hq[j]);
        for(int j=0;j<7;j++) fprintf(fxu, "%.9g,", hqd[j]);
        for(int j=0;j<7;j++) fprintf(fxu, "%.9g%s", hu[j], j<6?",":"\n");
    }
    fclose(fee); fclose(fxu);
    printf("wrote %s_eepos.traj + %s_traj.csv (%d steps, gravity=%.3f)\n", prefix.c_str(), prefix.c_str(), NSTEPS, gravity);
    return 0;
}
