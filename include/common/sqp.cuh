#pragma once
// Shared SQP driver. The linear-system backend is a policy object supplied by pcg/sqp.cuh
// (cooperative GBD-PCG on the GPU) or qdldl/sqp.cuh (QDLDL on the CPU). Everything else, the
// KKT formation, dz recovery, merit line search and rho schedule, is identical for both and
// lives here once. A backend provides:
//   static constexpr int kind;                       workspace backend tag (slot layout)
//   void setup(SqpWorkspace&, s, c, N, d_S, d_gamma, d_lambda);   its workspace slots, in a fixed order
//   void formSchur(s, c, N, d_G, d_C, d_g, d_c, d_S, d_gamma, rho);
//   void solve(d_lambda, iter_vec, exit_vec);        writes d_lambda; records iterations if it has any
//   template<class Dump> void dumpSchur(Dump);       DUMP_KKT only: backend-specific matrices
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>
#include <time.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "common/workspace.cuh"
#include "common/kkt.cuh"
#include "common/dz.cuh"
#include "merit.cuh"
#include "settings.cuh"
#ifdef DUMP_KKT
#include "utils/dump.hpp"
#endif

namespace mpcgpu {

// (linear-system iterations per SQP step, linear-system times, SQP wall time, SQP steps,
//  rho stayed below its maximum, linear-system exit flags)
using SqpStats = std::tuple<std::vector<int>, std::vector<double>, double, uint32_t, bool, std::vector<bool>>;

template <typename T, class Backend>
SqpStats sqpSolve(uint32_t state_size, uint32_t control_size, uint32_t knot_points, float timestep,
                  T* d_eePos_traj, T* d_lambda, T* d_xu, void* d_dynMem_const, Backend& backend,
                  T& rho, T rho_reset, T* d_xs_goal, SqpWorkspace* reuse) {
    if (state_size != 14 || control_size != 7 || knot_points != KNOT_POINTS || knot_points < 2)
        throw std::invalid_argument("MPCGPU requires iiwa14 dimensions and the compiled horizon");
    struct timespec sqp_solve_start, sqp_solve_end;
    gpuErrchk(cudaDeviceSynchronize());
#ifndef MPCGPU_CORRECTNESS
    clock_gettime(CLOCK_MONOTONIC, &sqp_solve_start);   // fresh workspace construction is part of the call
#endif
    std::unique_ptr<SqpWorkspace> owned;
    if (!reuse) { owned = std::make_unique<SqpWorkspace>(state_size, control_size, knot_points); reuse = owned.get(); }
    SqpWorkspace& workspace = *reuse;
    workspace.begin(state_size, control_size, knot_points, Backend::kind);

    std::vector<int> linsys_iter_vec;
    std::vector<bool> linsys_exit_vec;
    std::vector<double> linsys_time_vec;
    bool sqp_time_exit = 1;     // recorded, not a flag: cleared when rho exceeds its maximum

    const uint32_t states_sq = state_size*state_size;
    const uint32_t states_p_controls = state_size*control_size;
    const uint32_t controls_sq = control_size*control_size;
    const uint32_t states_s_controls = state_size + control_size;
    const uint32_t kkt_G = (states_sq+controls_sq)*knot_points - controls_sq;
    const uint32_t kkt_C = (states_sq+states_p_controls)*(knot_points-1);
    const uint32_t kkt_g = states_s_controls*knot_points - control_size;
    const uint32_t kkt_c = state_size*knot_points;
    const uint32_t dz_n = states_s_controls*knot_points - control_size;

    // line search over alpha = -2^-p, p = 0..7, one stream each
    const float mu = 10.0f;
    const uint32_t num_alphas = 8;
    T h_merit_news[num_alphas];
    void* ls_merit_kernel = (void*) ls_gato_compute_merit<T>;
    const size_t merit_smem_size = get_merit_smem_size<T>(state_size, control_size);
    T h_merit_initial, min_merit, alphafinal;
    uint32_t line_search_step = 0;
    auto& streams = workspace.streams;
    auto handle = workspace.handle;
    uint32_t sqp_iter = 0;
    T drho = 1.0, rho_factor = RHO_FACTOR, rho_max = RHO_MAX, rho_min = RHO_MIN;

    // Slot order is part of the workspace contract: shared slots first, then the backend's.
    T* d_G_dense = workspace.device<T>(kkt_G);
    T* d_C_dense = workspace.device<T>(kkt_C);
    T* d_g = workspace.device<T>(kkt_g);
    T* d_c = workspace.device<T>(kkt_c);
    T* d_Ginv_dense = d_G_dense;
    T* d_S = workspace.device<T>(3*states_sq*knot_points);
    T* d_gamma = workspace.device<T>(state_size*knot_points);
    gpuErrchk(cudaPeekAtLastError());
    T* d_dz = workspace.device<T>(dz_n);
    T* d_xs = workspace.device<T>(state_size);
    gpuErrchk(cudaMemcpy(d_xs, d_xu, state_size*sizeof(T), cudaMemcpyDeviceToDevice));
    T* d_merit_news = workspace.device<T>(8);
    T* d_merit_temp = workspace.device<T>(8*knot_points);
    T* d_merit_initial = workspace.device<T>(1);
    gpuErrchk(cudaMemset(d_merit_initial, 0, sizeof(T)));
    backend.setup(workspace, state_size, control_size, knot_points, d_S, d_gamma, d_lambda);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

#if TIME_LINSYS
    struct timespec linsys_start, linsys_end;
#endif
#if CONST_UPDATE_FREQ && !defined(MPCGPU_CORRECTNESS)
    struct timespec sqp_cur;
    auto sqpTimecheck = [&]() {
        clock_gettime(CLOCK_MONOTONIC, &sqp_cur);
        return time_delta_us_timespec(sqp_solve_start, sqp_cur) > SQP_MAX_TIME_US;
    };
#else
    auto sqpTimecheck = [&]() { return false; };
#endif

    // Deterministic merit: per-knot temp (no atomicAdd) + fixed-order reduce.
    compute_merit<T><<<knot_points, MERIT_THREADS, merit_smem_size>>>(
        state_size, control_size, knot_points, d_xu, d_eePos_traj, static_cast<T>(10), timestep,
        d_dynMem_const, d_merit_temp, d_xs_goal);
    reduce_merit<T><<<1, MERIT_THREADS>>>(knot_points, d_merit_temp, d_merit_initial);
    gpuErrchk(cudaMemcpyAsync(&h_merit_initial, d_merit_initial, sizeof(T), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaPeekAtLastError());

#ifdef DUMP_KKT
    // Capture one solve (DUMP_KKT_AT_SOLVE, counted across the run) into MPCGPU_DUMP_DIR for
    // offline analysis; tools/attribution uses this. Debug-only build flag.
#ifndef DUMP_KKT_AT_SOLVE
#define DUMP_KKT_AT_SOLVE 0
#endif
    static int dumped = 0;
    bool dump_this_solve = false;
    auto dump = [](const char* name, T* device, size_t count) { dumpDevice(name, device, count); };
#endif

    for (uint32_t sqpiter = 0; sqpiter < SQP_MAX_ITER; sqpiter++) {
        generate_kkt_submatrices<T, MPCGPU_INTEGRATOR><<<knot_points, KKT_THREADS, 2*get_kkt_smem_size<T>(state_size, control_size)>>>(
            state_size, control_size, knot_points, d_G_dense, d_C_dense, d_g, d_c, d_dynMem_const,
            timestep, d_eePos_traj, d_xs, d_xu, d_xs_goal);
        gpuErrchk(cudaPeekAtLastError());
        if (sqpTimecheck()) break;

        backend.formSchur(state_size, control_size, knot_points, d_G_dense, d_C_dense, d_g, d_c, d_S, d_gamma, rho);
        gpuErrchk(cudaPeekAtLastError());
#ifdef DUMP_KKT
        dump_this_solve = (dumped++ == DUMP_KKT_AT_SOLVE);
        if (dump_this_solve) {
            gpuErrchk(cudaDeviceSynchronize());
            // G = [Q_k R_k] per knot (last knot Q only); C = [A_k B_k] per non-terminal knot
            dump("mpc_G.bin", d_G_dense, kkt_G);
            dump("mpc_C.bin", d_C_dense, kkt_C);
            backend.dumpSchur(dump);
            dump("mpc_gamma.bin", d_gamma, state_size*knot_points);
            dump("mpc_g.bin", d_g, kkt_g);
            dump("mpc_c.bin", d_c, kkt_c);
            dump("mpc_lambda0.bin", d_lambda, state_size*knot_points);     // warm-start lambda
            dump("mpc_xu_pre.bin", d_xu, kkt_g);                           // warm-start trajectory
            printf("[DUMP_KKT] wrote MPCGPU_DUMP_DIR/mpc_{G,C,S,Pinv,gamma}.bin (state=%u ctrl=%u N=%u rho=%g)\n",
                   state_size, control_size, knot_points, (double)rho);
        }
#endif
        if (sqpTimecheck()) break;

#if TIME_LINSYS
        gpuErrchk(cudaDeviceSynchronize());
        if (sqpTimecheck()) break;
        clock_gettime(CLOCK_MONOTONIC, &linsys_start);
#endif
        backend.solve(d_lambda, linsys_iter_vec, linsys_exit_vec);
#if TIME_LINSYS
        gpuErrchk(cudaDeviceSynchronize());
        clock_gettime(CLOCK_MONOTONIC, &linsys_end);
        linsys_time_vec.push_back(time_delta_us_timespec(linsys_start, linsys_end));
#endif
        if (sqpTimecheck()) break;

        compute_dz(state_size, control_size, knot_points, d_Ginv_dense, d_C_dense, d_g, d_lambda, d_dz);
        gpuErrchk(cudaPeekAtLastError());
        if (sqpTimecheck()) break;
#ifdef DUMP_KKT
        if (dump_this_solve) {
            gpuErrchk(cudaDeviceSynchronize());
            dump("mpc_lambda1.bin", d_lambda, state_size*knot_points);
            dump("mpc_Ginv.bin", d_Ginv_dense, kkt_G);
            dump("mpc_dz.bin", d_dz, dz_n);
            printf("[DUMP_KKT] wrote MPCGPU_DUMP_DIR/mpc_{lambda1,Ginv,dz}.bin (post-solve)\n");
        }
#endif

        for (uint32_t p = 0; p < num_alphas; p++) {
            void* kernelArgs[] = {
                (void*)&state_size, (void*)&control_size, (void*)&knot_points, (void*)&d_xs, (void*)&d_xu,
                (void*)&d_eePos_traj, (void*)&mu, (void*)&timestep, (void*)&d_dynMem_const, (void*)&d_dz,
                (void*)&p, (void*)&d_merit_news, (void*)&d_merit_temp, (void*)&d_xs_goal};
            gpuErrchk(cudaLaunchCooperativeKernel(ls_merit_kernel, knot_points, MERIT_THREADS, kernelArgs,
                                                  get_merit_smem_size<T>(state_size, knot_points), streams[p]));
        }
        if (sqpTimecheck()) break;
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        cudaMemcpy(h_merit_news, d_merit_news, 8*sizeof(T), cudaMemcpyDeviceToHost);
        if (sqpTimecheck()) break;

        line_search_step = 0;
        min_merit = h_merit_initial;
        for (int i = 0; i < 8; i++) {
            if (h_merit_news[i] < min_merit) { min_merit = h_merit_news[i]; line_search_step = i; }
        }
#ifdef SQP_DEBUG
        {
            std::vector<T> hdz(dz_n), hg(dz_n);
            cudaMemcpy(hdz.data(), d_dz, dz_n*sizeof(T), cudaMemcpyDeviceToHost);
            cudaMemcpy(hg.data(), d_g, dz_n*sizeof(T), cudaMemcpyDeviceToHost);
            double nn = 0, gn = 0;
            for (auto v : hdz) nn += (double)v*(double)v;
            for (auto v : hg) gn += (double)v*(double)v;
            printf("[SQP_DEBUG] iter=%u ||dz||=%.4e ||d_g||=%.4e rho=%.3e merit_init=%.6e min_merit=%.6e step=%d\n",
                   sqp_iter, sqrt(nn), sqrt(gn), (double)rho, (double)h_merit_initial, (double)min_merit, line_search_step);
        }
#endif

        if (min_merit == h_merit_initial) {   // line search failure: raise rho, keep the iterate
            drho = max(drho*rho_factor, rho_factor);
            rho = max(rho*drho, rho_min);
            sqp_iter++;
            if (rho > rho_max) { sqp_time_exit = 0; rho = rho_reset; break; }
            continue;
        }
        alphafinal = -1.0 / (1 << line_search_step);   // step sign
        drho = min(drho/rho_factor, 1/rho_factor);
        rho = max(rho*drho, rho_min);
#if USE_DOUBLES
        cublasDaxpy(handle, dz_n, &alphafinal, d_dz, 1, d_xu, 1);
#else
        cublasSaxpy(handle, dz_n, &alphafinal, d_dz, 1, d_xu, 1);
#endif
        gpuErrchk(cudaPeekAtLastError());
        sqp_iter++;   // counted after the accepted update
#ifdef DUMP_KKT
        if (dump_this_solve) {
            gpuErrchk(cudaDeviceSynchronize());
            dump("mpc_xu_post.bin", d_xu, kkt_g);
            dump("mpc_goal.bin", d_eePos_traj, 6*knot_points);
            printf("[DUMP_KKT] wrote MPCGPU_DUMP_DIR/mpc_{xu_post,goal}.bin (accepted alpha=%g)\n", (double)alphafinal);
        }
#endif
        if (sqpTimecheck()) break;
        h_merit_initial = min_merit;
    }

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#ifdef MPCGPU_CORRECTNESS
    double sqp_solve_time = 0;
#else
    clock_gettime(CLOCK_MONOTONIC, &sqp_solve_end);
    double sqp_solve_time = time_delta_us_timespec(sqp_solve_start, sqp_solve_end);
#endif
    return std::make_tuple(linsys_iter_vec, linsys_time_vec, sqp_solve_time, sqp_iter, sqp_time_exit, linsys_exit_vec);
}

} // namespace mpcgpu
