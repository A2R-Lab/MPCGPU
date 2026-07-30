// Unit test for the terminal-knot tracking-cost gradient (regression test for the kkt.cuh
// terminal-reference smem-aliasing bug, fixed in 88c3853).
// Loads the COMMITTED solve-3000 dump inputs (tools/data/mpc_{xu_pre,goal}.bin — captured
// 2026-07-07 from a fair-config closed loop, post-fix; run from the repo root), then:
//   (A) replays gato_plant::trackingCostGradientAndHessian_lastblock exactly as kkt.cuh does;
//   (B) calls trackingCostGradHess directly on (x_63, ref_63) with a fresh arena.
// Prints s_qkp1 from both. Ground truth = N_COST * J_EEpos^T (p(q63) - ref63) via pinocchio
// at the CONTACT frame (the URDF's fixed EE joint, +4cm z off L7; GRiD >= e31f7bd includes
// the fixed-joint origin — the old L7-frame truth [0.0026, 0.1289, 0.0020, -0.3513, 0.0002,
// 0.0729, 0] matched the pre-fix origin-dropping codegen), w=50:
// [0.0053, 1.3757, 0.0034, -1.0444, 0.0003, 0.1088, 0.0000] — large because the dump's
// refs were tracked against the OLD EE point, so the terminal knot carries the ~4cm frame
// shift as EE error; the aliasing-bug signature is a gradient wrong by >>0.05 abs on top.
// To re-pin on new inputs: re-dump (see tools/run_gates.sh GATES_REDUMP), copy to
// tools/data/, recompute the truth with pinocchio (LOCAL_WORLD_ALIGNED contact-frame
// jacobian, position rows), update the print below.
//
// Build (repo root):
//   nvcc --compiler-options -Wall -O3 -DNDEBUG -arch=sm_120 -Iinclude -Iinclude/common -IGLASS \
//        -IGBD-PCG/include -Iqdldl/include tools/test_terminal_cost.cu -o tools/test_terminal_cost.exe -lcublas
#include <cstdio>
#include <cstdint>
#include <vector>
#include "settings.cuh"
#include "dynamics/iiwa/iiwa_eepos_plant.cuh"
#include "kkt.cuh"

using T = float;

__global__ void lastblock_kernel(T* d_xux, T* d_ref12, T* d_qkp1_out, void* d_dynMem){
    extern __shared__ T s_mem[];
    const uint32_t ss = 14, cs = 7;
    // mimic kkt.cuh generate_kkt_submatrices smem layout for the last block
    T *s_x_goal = s_mem;
    T *s_xux = s_x_goal + 2*ss;
    T *s_eePos_traj = s_xux + 2*ss + cs;
    T *s_Qk = s_eePos_traj + 2*6;   // mirrors the kkt.cuh fix (12 floats: refs for knots k AND k+1)
    T *s_Rk = s_Qk + ss*ss;
    T *s_qk = s_Rk + cs*cs;
    T *s_rk = s_qk + ss;
    T *s_end = s_rk + cs;
    T *s_Ak = s_end;
    T *s_Bk = s_Ak + ss*ss;
    T *s_Qkp1 = s_Bk + ss*cs;
    T *s_qkp1 = s_Qkp1 + ss*ss;
    T *s_integrator_error = s_qkp1 + ss;
    T *s_extra_temp = s_integrator_error + ss;

    for(int i = threadIdx.x; i < (int)(2*ss+cs); i += blockDim.x) s_xux[i] = d_xux[i];
    for(int i = threadIdx.x; i < 12; i += blockDim.x) s_eePos_traj[i] = d_ref12[i];
    __syncthreads();

    gato_plant::trackingCostGradientAndHessian_lastblock<T>(
        ss, cs, s_xux, s_eePos_traj, /*s_x_goal=*/nullptr,
        s_Qk, s_qk, s_Rk, s_rk, s_Qkp1, s_qkp1, s_extra_temp, d_dynMem);
    __syncthreads();
    for(int i = threadIdx.x; i < (int)ss; i += blockDim.x) d_qkp1_out[i] = s_qkp1[i];
}

__global__ void direct_kernel(T* d_x63, T* d_ref63, T* d_q_out, void* d_dynMem){
    extern __shared__ T s_mem[];
    const uint32_t ss = 14, cs = 7;
    T *s_x = s_mem;                       // terminal state
    T *s_ref = s_x + ss;                  // 6-wide terminal reference
    T *s_Q = s_ref + 6;
    T *s_q = s_Q + ss*ss;
    T *s_R = s_q + ss;
    T *s_r = s_R + cs*cs;
    T *s_temp = s_r + cs;                 // fresh arena, no offset

    for(int i = threadIdx.x; i < (int)ss; i += blockDim.x) s_x[i] = d_x63[i];
    for(int i = threadIdx.x; i < 6; i += blockDim.x) s_ref[i] = d_ref63[i];
    __syncthreads();

    gato_plant::trackingCostGradHess<T>(s_x, s_x, s_ref, /*s_x_goal=*/nullptr,
        s_Q, s_q, s_R, s_r, s_temp, (const grid::robotModel<T>*)d_dynMem,
        static_cast<T>(Q_COST), static_cast<T>(QD_COST), static_cast<T>(U_COST), static_cast<T>(Q_LIM_COST),
        static_cast<T>(VEL_LIM_COST), static_cast<T>(CTRL_LIM_COST), /*ee_weight=*/static_cast<T>(N_COST));
    __syncthreads();
    for(int i = threadIdx.x; i < (int)ss; i += blockDim.x) d_q_out[i] = s_q[i];
}

// (C) terminal call ONLY, but with the arena shifted +56 floats like the lastblock does
__global__ void shifted_kernel(T* d_x63, T* d_ref63, T* d_q_out, void* d_dynMem){
    extern __shared__ T s_mem[];
    const uint32_t ss = 14, cs = 7;
    T *s_x = s_mem; T *s_ref = s_x + ss;
    T *s_Q = s_ref + 6; T *s_q = s_Q + ss*ss; T *s_R = s_q + ss; T *s_r = s_R + cs*cs;
    T *s_temp = s_r + cs + 56;   // the lastblock's R_dummy/r_dummy carve
    for(int i = threadIdx.x; i < (int)ss; i += blockDim.x) s_x[i] = d_x63[i];
    for(int i = threadIdx.x; i < 6; i += blockDim.x) s_ref[i] = d_ref63[i];
    __syncthreads();
    gato_plant::trackingCostGradHess<T>(s_x, s_x, s_ref, nullptr, s_Q, s_q, s_R, s_r, s_temp,
        (const grid::robotModel<T>*)d_dynMem,
        (T)Q_COST, (T)QD_COST, (T)U_COST, (T)Q_LIM_COST, (T)VEL_LIM_COST, (T)CTRL_LIM_COST, (T)N_COST);
    __syncthreads();
    for(int i = threadIdx.x; i < (int)ss; i += blockDim.x) d_q_out[i] = s_q[i];
}

// (D) running call first, then terminal call on an UNSHIFTED fresh arena region
__global__ void seq_kernel(T* d_xux, T* d_ref12, T* d_q_out, void* d_dynMem){
    extern __shared__ T s_mem[];
    const uint32_t ss = 14, cs = 7;
    T *s_xux = s_mem; T *s_ref = s_xux + 2*ss + cs;
    T *s_Qk = s_ref + 12; T *s_qk = s_Qk + ss*ss; T *s_Rk = s_qk + ss; T *s_rk = s_Rk + cs*cs;
    T *s_Qkp1 = s_rk + cs; T *s_qkp1 = s_Qkp1 + ss*ss;
    T *s_temp = s_qkp1 + ss;
    for(int i = threadIdx.x; i < (int)(2*ss+cs); i += blockDim.x) s_xux[i] = d_xux[i];
    for(int i = threadIdx.x; i < 12; i += blockDim.x) s_ref[i] = d_ref12[i];
    __syncthreads();
    gato_plant::trackingCostGradHess<T>(s_xux, s_xux + ss, s_ref, nullptr, s_Qk, s_qk, s_Rk, s_rk, s_temp,
        (const grid::robotModel<T>*)d_dynMem,
        (T)Q_COST, (T)QD_COST, (T)U_COST, (T)Q_LIM_COST, (T)VEL_LIM_COST, (T)CTRL_LIM_COST, (T)EE_COST);
    __syncthreads();
    T *s_xkp1 = s_xux + ss + cs;
    gato_plant::trackingCostGradHess<T>(s_xkp1, s_xkp1, &s_ref[6], nullptr, s_Qkp1, s_qkp1, s_Rk, s_rk, s_temp,
        (const grid::robotModel<T>*)d_dynMem,
        (T)Q_COST, (T)QD_COST, (T)U_COST, (T)Q_LIM_COST, (T)VEL_LIM_COST, (T)CTRL_LIM_COST, (T)N_COST);
    __syncthreads();
    for(int i = threadIdx.x; i < (int)ss; i += blockDim.x) d_q_out[i] = s_qkp1[i];
}

static std::vector<T> loadf(const char* fn, size_t n){
    std::vector<T> v(n); FILE* f = fopen(fn, "rb");
    if(!f){ printf("missing %s\n", fn); exit(1); }
    if(fread(v.data(), sizeof(T), n, f) != n){ printf("short read %s\n", fn); exit(1); }
    fclose(f); return v;
}

int main(){
    const uint32_t ss = 14, cs = 7, N = 64;
    auto xu = loadf("tools/data/mpc_xu_pre.bin", (ss+cs)*N - cs);
    auto goal = loadf("tools/data/mpc_goal.bin", 6*N);

    grid::robotModel<T>* d_robotModel = grid::init_robotModel<T>();

    T *d_xux, *d_ref12, *d_out, *d_x63, *d_ref63;
    cudaMalloc(&d_xux, (2*ss+cs)*sizeof(T)); cudaMalloc(&d_ref12, 12*sizeof(T));
    cudaMalloc(&d_out, ss*sizeof(T)); cudaMalloc(&d_x63, ss*sizeof(T)); cudaMalloc(&d_ref63, 6*sizeof(T));
    // knot 62's xux = [x62 u62 x63]; ref12 = [ref62(6) ref63(6)]
    cudaMemcpy(d_xux, &xu[62*(ss+cs)], (2*ss+cs)*sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ref12, &goal[62*6], 12*sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x63, &xu[63*(ss+cs)], ss*sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ref63, &goal[63*6], 6*sizeof(T), cudaMemcpyHostToDevice);

    size_t smemA = 2*get_kkt_smem_size<T>(ss, cs);
    lastblock_kernel<<<1, KKT_THREADS, smemA>>>(d_xux, d_ref12, d_out, (void*)d_robotModel);
    cudaDeviceSynchronize();
    std::vector<T> outA(ss); cudaMemcpy(outA.data(), d_out, ss*sizeof(T), cudaMemcpyDeviceToHost);
    printf("(A) lastblock  s_qkp1 q-block: ");
    for(int i=0;i<7;i++) printf("% .4f ", (double)outA[i]); printf("\n");

    size_t smemB = sizeof(T)*(ss + 6 + ss*ss + ss + cs*cs + cs) + sizeof(T)*gato_plant::trackingCostGradHess_TempMemCt<T>();
    direct_kernel<<<1, KKT_THREADS, smemB>>>(d_x63, d_ref63, d_out, (void*)d_robotModel);
    cudaDeviceSynchronize();
    std::vector<T> outB(ss); cudaMemcpy(outB.data(), d_out, ss*sizeof(T), cudaMemcpyDeviceToHost);
    printf("(B) direct     s_q     q-block: ");
    for(int i=0;i<7;i++) printf("% .4f ", (double)outB[i]); printf("\n");
    shifted_kernel<<<1, KKT_THREADS, smemA>>>(d_x63, d_ref63, d_out, (void*)d_robotModel);
    cudaDeviceSynchronize();
    std::vector<T> outC(ss); cudaMemcpy(outC.data(), d_out, ss*sizeof(T), cudaMemcpyDeviceToHost);
    printf("(C) shifted+56 s_q     q-block: ");
    for(int i=0;i<7;i++) printf("% .4f ", (double)outC[i]); printf("\n");

    seq_kernel<<<1, KKT_THREADS, smemA>>>(d_xux, d_ref12, d_out, (void*)d_robotModel);
    cudaDeviceSynchronize();
    std::vector<T> outD(ss); cudaMemcpy(outD.data(), d_out, ss*sizeof(T), cudaMemcpyDeviceToHost);
    printf("(D) run+term unshifted q-block: ");
    for(int i=0;i<7;i++) printf("% .4f ", (double)outD[i]); printf("\n");

    printf("(pin truth, w=50):              0.0053  1.3757  0.0034 -1.0444  0.0003  0.1088  0.0000\n");
    return 0;
}
