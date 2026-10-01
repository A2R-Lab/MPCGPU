#pragma once
// GBD-PCG backend for the shared SQP driver (common/sqp.cuh): the Schur system is formed as
// block-tridiagonal strips with its preconditioner and solved by the cooperative grid-wide PCG.
#include <algorithm>
#include <cstdint>
#include <vector>
#include "common/sqp.cuh"
#include "linsys_setup.cuh"
#include "gpu_pcg.cuh"

namespace mpcgpu {

template <typename T>
struct PcgBackend {
    static constexpr int kind = 0;
    pcg_config<T>& config;
    uint32_t knot_points = 0;
    size_t smem = 0;
    T *d_S = nullptr, *d_gamma = nullptr, *d_lambda = nullptr;
    T *d_Pinv = nullptr, *d_r = nullptr, *d_p = nullptr, *d_v_temp = nullptr, *d_eta_new_temp = nullptr;
    uint32_t* d_pcg_iters = nullptr;
    bool* d_pcg_exit = nullptr;
    void* args[13];

    explicit PcgBackend(pcg_config<T>& c) : config(c) {}

    void setup(SqpWorkspace& workspace, uint32_t state_size, uint32_t, uint32_t knots, T* S, T* gamma, T* lambda) {
        knot_points = knots; d_S = S; d_gamma = gamma; d_lambda = lambda;
        d_Pinv = workspace.device<T>(3*state_size*state_size*knots);
        d_r = workspace.device<T>(state_size*knots);
        d_p = workspace.device<T>(state_size*knots);
        d_v_temp = workspace.device<T>(knots);
        d_eta_new_temp = workspace.device<T>(knots);
        d_pcg_iters = workspace.device<uint32_t>(1);
        d_pcg_exit = workspace.device<bool>(1);
        smem = pcgSharedMemSize<T>(state_size, knots);
        void* a[13] = {(void*)&d_S, (void*)&d_Pinv, (void*)&d_gamma, (void*)&d_lambda, (void*)&d_r, (void*)&d_p,
                       (void*)&d_v_temp, (void*)&d_eta_new_temp, (void*)&d_pcg_iters, (void*)&d_pcg_exit,
                       (void*)&config.pcg_max_iter, (void*)&config.pcg_exit_tol, (void*)&config.pcg_rel_tol};
        std::copy(a, a + 13, args);
    }

    void formSchur(uint32_t state_size, uint32_t control_size, uint32_t knots, T* d_G, T* d_C, T* d_g, T* d_c,
                   T* S, T* gamma, T rho) {
        form_schur_system<T>(state_size, control_size, knots, d_G, d_C, d_g, d_c, S, d_Pinv, gamma, rho);
    }

    void solve(T*, std::vector<int>& iter_vec, std::vector<bool>& exit_vec) {
        gpuErrchk(cudaLaunchCooperativeKernel((void*) pcg<T, STATE_SIZE, KNOT_POINTS>, knot_points, PCG_NUM_THREADS, args, smem));
        uint32_t iterations; bool exit;
        gpuErrchk(cudaMemcpy(&iterations, d_pcg_iters, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(&exit, d_pcg_exit, sizeof(bool), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaPeekAtLastError());
        iter_vec.push_back(iterations);
        exit_vec.push_back(exit);
    }

#ifdef DUMP_KKT
    template <class Dump> void dumpSchur(Dump dump) {
        const size_t strips = 3*(size_t)STATE_SIZE*STATE_SIZE*knot_points;   // [L|D|R] per knot
        dump("mpc_S.bin", d_S, strips);
        dump("mpc_Pinv.bin", d_Pinv, strips);
    }
#endif
};

} // namespace mpcgpu

template <typename T>
auto sqpSolvePcg(const uint32_t state_size, const uint32_t control_size, const uint32_t knot_points, float timestep,
                 T* d_eePos_traj, T* d_lambda, T* d_xu, void* d_dynMem_const, pcg_config<T>& config, T& rho, T rho_reset,
                 T* d_xs_goal = nullptr, mpcgpu::SqpWorkspace* reuse = nullptr) {
    mpcgpu::PcgBackend<T> backend(config);
    return mpcgpu::sqpSolve<T>(state_size, control_size, knot_points, timestep, d_eePos_traj, d_lambda, d_xu,
                               d_dynMem_const, backend, rho, rho_reset, d_xs_goal, reuse);
}
