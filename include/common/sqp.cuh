#pragma once
// Shared SQP driver. The linear-system backend is a policy object supplied by pcg/sqp.cuh
// (cooperative GBD-PCG on the GPU) or qdldl/sqp.cuh (QDLDL on the CPU). Everything else, the
// KKT formation, dz recovery, merit line search and rho schedule, is identical for both and
// lives here once. A backend provides:
//   static constexpr int kind;                       workspace backend tag (slot layout)
//   static constexpr bool graphable;                 its solve is device-only work (capturable)
//   void setup(SqpWorkspace&, s, c, N, d_S, d_gamma, d_lambda);   its workspace slots, in a fixed order
//   void formSchur(s, c, N, d_G, d_C, d_g, d_c, d_S, d_gamma, rho, d_rho, stream);
//   void solve(d_lambda, stream);                    writes d_lambda (enqueued on `stream`)
//   void copyStats(stream);                          its statistics for this solve become readable after the next host sync
//   void graphKey(std::vector<uint64_t>&);           append every by-value launch argument its graphs freeze
//   void record(iter_vec, exit_vec);                 after a host sync: push the copied statistics
//   template<class Dump> void dumpSchur(Dump);       DUMP_KKT only: backend-specific matrices
//
// Launch structure. All solver work runs on the workspace's main stream; the eight line-search
// merits fork to the workspace's side streams and join back; the initial merit runs on a ninth
// stream alongside the KKT formation. One SQP step is three segments — pre (KKT, Schur),
// linsys (the solve), post (dz, line-search merits) — that the pcg
// backend records once per caller-provided (reused) workspace as CUDA graphs and replays
// (MPCGPU_GRAPH, default on; a per-call workspace, DUMP_KKT and the host-side qdldl solve launch
// the same segments directly).
// With a caller-provided (reused) workspace the kernels store the words the host needs (eight
// line-search merits, the initial merit, the PCG iteration count and exit flag) straight into
// mapped page-locked host memory, so there is no device-to-host copy anywhere in a step; a
// per-call workspace does not page-lock (too slow per call) and copies them back instead. The host synchronizes once per SQP step, to read the
// merits and pick the step, and once at the end of the solve (the timing boundary), where the
// linear-system statistics are read back. TIME_LINSYS builds add the paper's two device syncs
// around the linear-system segment so that metric keeps its definition.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
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

#ifndef MPCGPU_GRAPH
#define MPCGPU_GRAPH 1
#endif
#ifdef DUMP_KKT          // the dumps synchronize and read between launches; replay would hide them
#undef MPCGPU_GRAPH
#define MPCGPU_GRAPH 0
#endif

namespace mpcgpu {

// (linear-system iterations per SQP step, linear-system times, SQP wall time, SQP steps,
//  rho stayed below its maximum, linear-system exit flags)
using SqpStats = std::tuple<std::vector<int>, std::vector<double>, double, uint32_t, bool, std::vector<bool>>;

// Bit pattern of a by-value launch argument, for the graph key.
template <class V> uint64_t graph_key_bits(const V& value) {
    static_assert(sizeof(V) <= sizeof(uint64_t), "key values are at most 64 bits");
    uint64_t bits = 0; std::memcpy(&bits, &value, sizeof(V)); return bits;
}

// Workspace event slots used by the driver.
enum : size_t { kEventFork = 0, kEventJoin = 1 /* +8 */, kEventInitial = 9 };

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
    workspace.owned_by_caller = !owned;

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
    void* ls_merit_kernel = (void*) ls_gato_compute_merit<T>;
    const size_t merit_smem_size = get_merit_smem_size<T>(state_size, control_size);
    T min_merit, alphafinal;
    uint32_t line_search_step = 0;
    auto& streams = workspace.streams;
    cudaStream_t ms = workspace.main_stream();      // every solver launch; cuBLAS is bound to it
    cudaStream_t side = workspace.side_stream();    // the initial merit, alongside the KKT formation
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
    gpuErrchk(cudaMemcpyAsync(d_xs, d_xu, state_size*sizeof(T), cudaMemcpyDeviceToDevice, ms));
    T* d_merit_temp = workspace.device<T>(8*knot_points);
    T* d_rho = workspace.device<T>(1);                       // the Schur kernels read rho from here
    // Pinned/mapped slots first (shared), then the backend's.
    T* h_rho = workspace.pinned<T>(SQP_MAX_ITER);           // one slot per step: the copy may still be in flight
    T *d_merit_news, *d_merit_initial;                       // kernels store these into host memory
    T* h_merit_news = workspace.mapped<T>(num_alphas, &d_merit_news);
    T* h_merit_initial_slot = workspace.mapped<T>(1, &d_merit_initial);
    backend.setup(workspace, state_size, control_size, knot_points, d_S, d_gamma, d_lambda);
    gpuErrchk(cudaPeekAtLastError());
    T h_merit_initial;

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

    // Initial merit on the side stream, overlapping the first KKT formation. Deterministic:
    // per-knot temp (no atomicAdd) + fixed-order reduce. It shares d_merit_temp with the
    // line-search merits, so the side stream joins the main stream before those launch.
    compute_merit<T><<<knot_points, MERIT_THREADS, merit_smem_size, side>>>(
        state_size, control_size, knot_points, d_xu, d_eePos_traj, static_cast<T>(10), timestep,
        d_dynMem_const, d_merit_temp, d_xs_goal);
    reduce_merit<T><<<1, MERIT_THREADS, 0, side>>>(knot_points, d_merit_temp, d_merit_initial);
    if (workspace.copies_needed())
        gpuErrchk(cudaMemcpyAsync(h_merit_initial_slot, d_merit_initial, sizeof(T), cudaMemcpyDeviceToHost, side));
    gpuErrchk(cudaEventRecord(workspace.event(kEventInitial), side));
    gpuErrchk(cudaPeekAtLastError());
    bool merit_initial_read = false;
    // Host bookkeeping must follow actual launches, never graph capture: capture
    // executes host code once, whereas replay executes only device operations.
    bool linsys_pending = false;

    // The three segments of one SQP step. Captured once per workspace into graphs (pcg) or
    // launched directly (qdldl, DUMP_KKT). Pointers and sizes are fixed per workspace+caller.
    auto seg_pre = [&](cudaStream_t s) {
        generate_kkt_submatrices<T, MPCGPU_INTEGRATOR><<<knot_points, KKT_THREADS, 2*get_kkt_smem_size<T>(state_size, control_size), s>>>(
            state_size, control_size, knot_points, d_G_dense, d_C_dense, d_g, d_c, d_dynMem_const,
            timestep, d_eePos_traj, d_xs, d_xu, d_xs_goal);
        backend.formSchur(state_size, control_size, knot_points, d_G_dense, d_C_dense, d_g, d_c, d_S, d_gamma,
                          rho, d_rho, s);
    };
    auto seg_lin = [&](cudaStream_t s) {
        backend.solve(d_lambda, s);
        backend.copyStats(s);
    };
    auto seg_post = [&](cudaStream_t s) {
        compute_dz(state_size, control_size, knot_points, d_Ginv_dense, d_C_dense, d_g, d_lambda, d_dz, s);
        gpuErrchk(cudaEventRecord(workspace.event(kEventFork), s));
        for (uint32_t p = 0; p < num_alphas; p++) {
            void* kernelArgs[] = {
                (void*)&state_size, (void*)&control_size, (void*)&knot_points, (void*)&d_xs, (void*)&d_xu,
                (void*)&d_eePos_traj, (void*)&mu, (void*)&timestep, (void*)&d_dynMem_const, (void*)&d_dz,
                (void*)&p, (void*)&d_merit_news, (void*)&d_merit_temp, (void*)&d_xs_goal};
            gpuErrchk(cudaStreamWaitEvent(streams[p], workspace.event(kEventFork), 0));
            gpuErrchk(cudaLaunchCooperativeKernel(ls_merit_kernel, knot_points, MERIT_THREADS, kernelArgs,
                                                  get_merit_smem_size<T>(state_size, knot_points), streams[p]));
            gpuErrchk(cudaEventRecord(workspace.event(kEventJoin + p), streams[p]));
        }
        for (uint32_t p = 0; p < num_alphas; p++)
            gpuErrchk(cudaStreamWaitEvent(s, workspace.event(kEventJoin + p), 0));
        if (workspace.copies_needed())
            gpuErrchk(cudaMemcpyAsync(h_merit_news, d_merit_news, num_alphas*sizeof(T), cudaMemcpyDeviceToHost, s));
    };
    // Graphs pay off only when the workspace outlives the call: a per-call workspace would capture
    // three graphs per solve, so it launches the segments directly.
    const bool use_graph = MPCGPU_GRAPH && Backend::graphable && workspace.owned_by_caller;
    if (use_graph) {
        // Everything the captured launches hold by value: caller pointers, the timestep, the
        // backend's parameters (the sim warm-starts with tighter PCG tolerances, then switches).
        std::vector<uint64_t> key = {graph_key_bits(d_eePos_traj), graph_key_bits(d_lambda), graph_key_bits(d_xu),
                                     graph_key_bits(d_dynMem_const), graph_key_bits(d_xs_goal), graph_key_bits(timestep)};
        backend.graphKey(key);
        if (!workspace.graphsMatch(key)) {
            workspace.destroyGraphs();
            workspace.captureGraph(0, ms, [&]{ seg_pre(ms); });
            workspace.captureGraph(1, ms, [&]{ seg_lin(ms); });
            workspace.captureGraph(2, ms, [&]{ seg_post(ms); });
            workspace.setGraphKey(key);
        }
    }

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
        h_rho[sqpiter] = rho;
        gpuErrchk(cudaMemcpyAsync(d_rho, &h_rho[sqpiter], sizeof(T), cudaMemcpyHostToDevice, ms));
        if (use_graph) workspace.launchGraph(0, ms); else seg_pre(ms);
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
        // The paper's linear-system time: host wall between a device sync before and after the solve
        // (ICRA Figures 4/5 and the site compare against it). These two syncs exist only in
        // TIME_LINSYS builds; everything else in the step stays stream-ordered.
        gpuErrchk(cudaDeviceSynchronize());
        if (sqpTimecheck()) break;
        clock_gettime(CLOCK_MONOTONIC, &linsys_start);
#endif
        if (use_graph) workspace.launchGraph(1, ms); else seg_lin(ms);
        linsys_pending = true;
#if TIME_LINSYS
        gpuErrchk(cudaDeviceSynchronize());
        clock_gettime(CLOCK_MONOTONIC, &linsys_end);
        linsys_time_vec.push_back(time_delta_us_timespec(linsys_start, linsys_end));
#endif
        if (sqpTimecheck()) break;

        if (sqpiter == 0) gpuErrchk(cudaStreamWaitEvent(ms, workspace.event(kEventInitial), 0));   // d_merit_temp handoff
        if (use_graph) workspace.launchGraph(2, ms); else seg_post(ms);
        gpuErrchk(cudaPeekAtLastError());
#ifdef DUMP_KKT
        if (dump_this_solve) {
            gpuErrchk(cudaDeviceSynchronize());
            dump("mpc_lambda1.bin", d_lambda, state_size*knot_points);
            dump("mpc_Ginv.bin", d_Ginv_dense, kkt_G);
            dump("mpc_dz.bin", d_dz, dz_n);
            printf("[DUMP_KKT] wrote MPCGPU_DUMP_DIR/mpc_{lambda1,Ginv,dz}.bin (post-solve)\n");
        }
#endif
        if (sqpTimecheck()) break;
        // The decision point: the only host wait inside a step.
        gpuErrchk(cudaStreamSynchronize(ms));
        backend.record(linsys_iter_vec, linsys_exit_vec);
        linsys_pending = false;
        if (!merit_initial_read) { h_merit_initial = *h_merit_initial_slot; merit_initial_read = true; }
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
    if (linsys_pending) backend.record(linsys_iter_vec, linsys_exit_vec); // budget interrupted a launched step
    return std::make_tuple(linsys_iter_vec, linsys_time_vec, sqp_solve_time, sqp_iter, sqp_time_exit, linsys_exit_vec);
}

} // namespace mpcgpu
