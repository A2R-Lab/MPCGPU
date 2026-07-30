// ISOLATION TEST: does ONE sqpSolvePcg respond to the EE cost? Bypasses the closed-loop harness
// (mpcsim). Loads the hold warm-start d_xu + the fig8 goal WINDOW (first knot_points goals, which move
// away from center over the horizon), does ONE solve, then FKs every knot of the SOLVED d_xu and reports
// how well the solved trajectory tracks the goal window (mean |EE_k - goal_k|) vs the pre-solve hold.
// If the solved trajectory tracks the window (and the number changes with EE_COST at build time), the
// SOLVER+cost are fine => the bug is in mpcsim's control application. If it holds (== pre-solve) and is
// EE_COST-invariant, the cost->KKT->solve wiring is broken.  Run from repo root.  arg1=prefix.
#include <cstdio>
#include <vector>
#include <cmath>
#include "mpcsim.cuh"
#include "dynamics/rbd_plant.cuh"
#include "settings.cuh"
#include "utils/experiment.cuh"
#include "gpu_pcg.cuh"

template<typename T>
__global__ void k_fk_knot(T* d_ee, const T* d_xu, int knot, int xu_stride, void* d_rm){
    __shared__ T s_q[7], s_ee[6];
    for(int i=threadIdx.x;i<7;i+=blockDim.x) s_q[i]=d_xu[knot*xu_stride+i];
    __syncthreads();
    grid::end_effector_pose_device_EE<T>(s_ee, s_q, (grid::robotModel<T>*)d_rm);
    __syncthreads();
    for(int i=threadIdx.x;i<6;i+=blockDim.x) d_ee[i]=s_ee[i];
}

int main(int argc, char** argv){
    using T = linsys_t;
    constexpr uint32_t state_size = grid::NUM_JOINTS*2;    // 14
    constexpr uint32_t control_size = grid::NUM_JOINTS;    // 7
    constexpr uint32_t knot_points = KNOT_POINTS;
    const uint32_t xu_stride = state_size + control_size;  // 21
    const uint32_t traj_len = xu_stride*knot_points - control_size;
    const float timestep = TIMESTEP;
    std::string prefix = (argc>1)?argv[1]:"examples/trajfiles/0_0";

    auto ee2d = readCSVToVecVec<T>((prefix+"_eepos.traj").c_str());
    auto xu2d = readCSVToVecVec<T>((prefix+"_traj.csv").c_str());
    std::vector<T> h_ee, h_xu;
    for(auto&v:ee2d) h_ee.insert(h_ee.end(),v.begin(),v.end());
    for(auto&v:xu2d) h_xu.insert(h_xu.end(),v.begin(),v.end());

    // goal window = first knot_points goals (6-wide); warm-start d_xu = first traj_len of the hold
    T *d_eePos_goal,*d_xu,*d_lambda,*d_ee;
    gpuErrchk(cudaMalloc(&d_eePos_goal, 6*knot_points*sizeof(T)));
    gpuErrchk(cudaMemcpy(d_eePos_goal, h_ee.data(), 6*knot_points*sizeof(T), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc(&d_xu, traj_len*sizeof(T)));
    gpuErrchk(cudaMemcpy(d_xu, h_xu.data(), traj_len*sizeof(T), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc(&d_lambda, state_size*knot_points*sizeof(T)));
    gpuErrchk(cudaMemset(d_lambda, 0, state_size*knot_points*sizeof(T)));
    gpuErrchk(cudaMalloc(&d_ee, 6*sizeof(T)));
    void* d_rm = (void*)grid::init_robotModel<T>();
    size_t fk_smem = grid::END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_COUNT*sizeof(T);

    auto window_err = [&](const char* tag){
        double tot=0; int cnt=0;
        std::vector<T> he(6);
        for(uint32_t k=0;k<knot_points;k++){
            k_fk_knot<T><<<1,32,fk_smem>>>(d_ee, d_xu, k, xu_stride, d_rm);
            gpuErrchk(cudaDeviceSynchronize());
            gpuErrchk(cudaMemcpy(he.data(), d_ee, 6*sizeof(T), cudaMemcpyDeviceToHost));
            double e=0; for(int i=0;i<3;i++){ double d=he[i]-h_ee[k*6+i]; e+=d*d; }
            tot+=std::sqrt(e); cnt++;
            if(k==0||k==knot_points/2||k==knot_points-1)
                printf("  [%s] knot %2u EE=[%.4f %.4f %.4f] goal=[%.4f %.4f %.4f]\n",
                       tag,k,he[0],he[1],he[2], h_ee[k*6],h_ee[k*6+1],h_ee[k*6+2]);
        }
        printf("  [%s] mean |EE_k - goal_k| over window = %.6f\n", tag, tot/cnt);
    };

    printf("EE_COST=%.1f N_COST=%.1f  KNOT_POINTS=%u SQP_MAX_ITER=%d\n",(double)EE_COST,(double)N_COST,knot_points,SQP_MAX_ITER);
    window_err("pre ");

    pcg_config<T> config;
    config.pcg_exit_tol = 1e-6; config.pcg_rel_tol = PCG_RES_TOL; config.pcg_max_iter = PCG_MAX_ITER;
#ifdef TIGHT_PCG
    config.pcg_exit_tol = 1e-11; config.pcg_rel_tol = 1e-11; config.pcg_max_iter = 10000;  // diagnostic: fully converge the linear solve
#endif
    T rho = RHO_INIT;
    // snapshot d_xu to measure the solve's displacement
    std::vector<T> before(traj_len); gpuErrchk(cudaMemcpy(before.data(), d_xu, traj_len*sizeof(T), cudaMemcpyDeviceToHost));
    sqpSolvePcg<T>(state_size, control_size, knot_points, timestep, d_eePos_goal, d_lambda, d_xu, d_rm, config, rho, RHO_INIT);
    std::vector<T> after(traj_len);  gpuErrchk(cudaMemcpy(after.data(), d_xu, traj_len*sizeof(T), cudaMemcpyDeviceToHost));
    double disp=0; for(uint32_t i=0;i<traj_len;i++){ double d=after[i]-before[i]; disp+=d*d; }
    printf("|| d_xu_after - d_xu_before || = %.6e\n", std::sqrt(disp));
    window_err("post");
    return 0;
}
