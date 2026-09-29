// ICRA 2024 five-goal pick-and-place circuit on the maintained solver stack.
// Build: python tools/build.py icra-pcg --knots 64   (or icra-qdldl). Run from the repository root:
//   icra_pick_place [--reference PREFIX] [--tol EPS] [--trials K] [--out DIR]
// Writes DIR/summary.json, DIR/samples.csv (EE and goal per reference offset) and DIR/updates.csv
// (state, applied control and SQP iterations per control update). tools/icra_report.py checks
// goal completion and limits. The protocol and its differences from 2024 are documented in
// docs/icra-replication.md; correctness builds use a fixed simulated control period and no timers.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>
#include "mpcsim.cuh"
#include "dynamics/rbd_plant.cuh"
#include "settings.cuh"
#include "utils/experiment.cuh"
#include "gpu_pcg.cuh"
#include "utils/trajectory.hpp"

namespace {
constexpr uint32_t state_size = grid::NUM_JOINTS*2;
constexpr uint32_t control_size = grid::NUM_JOINTS;
constexpr uint32_t knot_points = KNOT_POINTS;

// Default PCG exit tolerance on eta = r^T Pinv r: the middle entry of each 2024 tolerance sweep.
// For N>=128 that is 1e-4, the tolerance the paper names for its N=128 results (Figures 4 and 5).
float default_tolerance() { return knot_points == 32 ? 5e-6f : (knot_points == 64 ? 5e-5f : 1e-4f); }

uint64_t state_hash(const std::vector<linsys_t>& states) {
    uint64_t hash = 14695981039346656037ull;
    for (const linsys_t& value : states) {
        const auto* bytes = reinterpret_cast<const unsigned char*>(&value);
        for (size_t i = 0; i < sizeof(value); ++i) { hash ^= bytes[i]; hash *= 1099511628211ull; }
    }
    return hash;
}

template<class V> double mean(const V& values) {
    return values.empty() ? 0.0 : std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}
}

int main(int argc, char** argv) try {
    std::string reference_prefix = "examples/icra/pick_place";
    std::string out_dir = std::string("tmp/icra/") + (LINSYS_SOLVE ? "pcg" : "qdldl") + "-N" + std::to_string(knot_points);
    float tolerance = default_tolerance();
    int trials = 1;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (i + 1 >= argc) throw std::invalid_argument("Missing value for " + arg);
        if (arg == "--reference") reference_prefix = argv[++i];
        else if (arg == "--out") out_dir = argv[++i];
        else if (arg == "--tol") tolerance = std::stof(argv[++i]);
        else if (arg == "--trials") trials = std::stoi(argv[++i]);
        else throw std::invalid_argument("Unknown argument " + arg);
    }
    if (trials < 1 || !std::isfinite(tolerance) || tolerance < 0) throw std::invalid_argument("Invalid --trials or --tol");

    mpcgpu::Reference<linsys_t> reference(reference_prefix, knot_points, state_size, control_size);
    std::filesystem::create_directories(out_dir);   // timing builds also dump per-trial data here
    linsys_t *d_eePos_traj, *d_xu_traj, *d_xs;
    gpuErrchk(cudaMalloc(&d_eePos_traj, reference.ee.size()*sizeof(linsys_t)));
    gpuErrchk(cudaMemcpy(d_eePos_traj, reference.ee.data(), reference.ee.size()*sizeof(linsys_t), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc(&d_xu_traj, reference.xu.size()*sizeof(linsys_t)));
    gpuErrchk(cudaMemcpy(d_xu_traj, reference.xu.data(), reference.xu.size()*sizeof(linsys_t), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc(&d_xs, state_size*sizeof(linsys_t)));

    std::vector<uint64_t> hashes;
    mpcgpu::TrackingLog<linsys_t> log;
    for (int trial = 0; trial < trials; ++trial) {
        mpcgpu::TrackingLog<linsys_t> current;
        gpuErrchk(cudaMemcpy(d_xs, reference.xu.data(), state_size*sizeof(linsys_t), cudaMemcpyHostToDevice));
        simulateMPC<linsys_t, toplevel_return_type>(state_size, control_size, knot_points,
            static_cast<uint32_t>(reference.steps), TIMESTEP, d_eePos_traj, d_xu_traj, d_xs,
            0, 0, trial, tolerance, out_dir + "/trial", &current);
        hashes.push_back(state_hash(current.states));
        if (trial == 0) log = std::move(current);
    }
    gpuErrchk(cudaFree(d_eePos_traj));
    gpuErrchk(cudaFree(d_xu_traj));
    gpuErrchk(cudaFree(d_xs));

    const size_t samples = log.ee.size() / 3;
    if (log.states.size() != (log.control_updates + 1) * state_size || log.controls.size() != log.control_updates * control_size
        || log.sqp_iters.size() != log.control_updates || log.ee_goal.size() != log.ee.size())
        throw std::runtime_error("Incomplete tracking log");
    for (linsys_t v : log.states) if (!std::isfinite(v)) throw std::runtime_error("Nonfinite state");
    for (linsys_t v : log.controls) if (!std::isfinite(v)) throw std::runtime_error("Nonfinite control");
    std::vector<double> errors(samples);
    for (size_t k = 0; k < samples; ++k) {
        double sum = 0;
        for (int a = 0; a < 3; ++a) { double d = log.ee[3*k+a] - log.ee_goal[3*k+a]; sum += d*d; }
        errors[k] = std::sqrt(sum);
        if (!std::isfinite(errors[k])) throw std::runtime_error("Nonfinite tracking error");
    }
    if (samples == 0) throw std::runtime_error("No tracking samples");

    {
        std::ofstream csv(out_dir + "/samples.csv");
        csv << "offset,time_s,ee_x,ee_y,ee_z,goal_x,goal_y,goal_z\n";
        csv.precision(9);
        for (size_t k = 0; k < samples; ++k) {
            csv << k << ',' << (k + 1) * TIMESTEP;
            for (int a = 0; a < 3; ++a) csv << ',' << log.ee[3*k+a];
            for (int a = 0; a < 3; ++a) csv << ',' << log.ee_goal[3*k+a];
            csv << '\n';
        }
    }
    {
        std::ofstream csv(out_dir + "/updates.csv");
        csv << "update,time_s";
        for (uint32_t i = 0; i < state_size/2; ++i) csv << ",q" << i;
        for (uint32_t i = 0; i < state_size/2; ++i) csv << ",qd" << i;
        for (uint32_t i = 0; i < control_size; ++i) csv << ",u" << i;
        csv << ",sqp_iters,rho_exit\n";
        csv.precision(9);
        for (uint32_t k = 0; k < log.control_updates; ++k) {
            csv << k << ',' << (k + 1) * SIMULATION_PERIOD * 1e-6;
            for (uint32_t i = 0; i < state_size; ++i) csv << ',' << log.states[(k+1)*state_size + i];
            for (uint32_t i = 0; i < control_size; ++i) csv << ',' << log.controls[k*control_size + i];
            csv << ',' << log.sqp_iters[k] << ',' << int(!log.sqp_exits[k]) << '\n';
        }
    }
    const size_t pcg_capped = std::count(log.linsys_exits.begin(), log.linsys_exits.end(), true);
    const size_t rho_exits = std::count(log.sqp_exits.begin(), log.sqp_exits.end(), false);
    const bool deterministic = std::all_of(hashes.begin(), hashes.end(), [&](uint64_t h){ return h == hashes[0]; });
    FILE* json = fopen((out_dir + "/summary.json").c_str(), "w");
    if (!json) throw std::runtime_error("Cannot write summary.json in " + out_dir);
    fprintf(json, "{\n  \"task\": \"icra2024_pick_place\",\n  \"reference\": \"%s\",\n", reference_prefix.c_str());
    fprintf(json, "  \"config\": {\"backend\": \"%s\", \"knot_points\": %u, \"timestep_s\": %.9g, "
                  "\"control_period_us\": %d, \"sqp_max_iter\": %d, \"pcg_max_iter\": %d, \"pcg_exit_tol\": %.9g, "
                  "\"ee_cost\": %.9g, \"terminal_ee_cost\": %.9g, \"qd_cost\": %.9g, \"u_cost\": %.9g, "
                  "\"rho_init\": %.9g, \"rho_max\": %.9g, \"warmup_reset\": %d, \"reference_tail_fill\": %d, "
                  "\"timers\": %s},\n",
            LINSYS_SOLVE ? "pcg" : "qdldl", knot_points, (double)TIMESTEP, SIMULATION_PERIOD, SQP_MAX_ITER,
            (int)PCG_MAX_ITER, LINSYS_SOLVE ? (double)tolerance : -1.0, (double)EE_COST, (double)N_COST,
            (double)QD_COST, (double)U_COST, (double)RHO_INIT, (double)RHO_MAX, WARMUP_RESET, REFERENCE_TAIL_FILL,
#ifdef MPCGPU_CORRECTNESS
            "false"
#else
            "true"
#endif
            );
    fprintf(json, "  \"trials\": %d,\n  \"deterministic\": %s,\n  \"state_hashes\": [", trials, deterministic ? "true" : "false");
    for (size_t i = 0; i < hashes.size(); ++i) fprintf(json, "%s\"%016llx\"", i ? ", " : "", (unsigned long long)hashes[i]);
    fprintf(json, "],\n  \"reference_rows\": %zu,\n  \"offsets\": %zu,\n  \"control_updates\": %u,\n",
            reference.steps, samples, log.control_updates);
    fprintf(json, "  \"l2_error_m\": {\"mean\": %.9g, \"max\": %.9g, \"final\": %.9g},\n",
            mean(errors), *std::max_element(errors.begin(), errors.end()), errors.back());
    fprintf(json, "  \"sqp\": {\"mean_iters\": %.6g, \"max_iters\": %u, \"rho_exits\": %zu},\n",
            mean(log.sqp_iters), *std::max_element(log.sqp_iters.begin(), log.sqp_iters.end()), rho_exits);
    fprintf(json, "  \"linsys\": {\"solves\": %zu, \"mean_pcg_iters\": %.6g, \"max_pcg_iters\": %d, \"pcg_max_iter_exits\": %zu}\n}\n",
            LINSYS_SOLVE ? log.linsys_iters.size() : 0, mean(log.linsys_iters),
            log.linsys_iters.empty() ? 0 : *std::max_element(log.linsys_iters.begin(), log.linsys_iters.end()), pcg_capped);
    fclose(json);
    printf("RESULT offsets=%zu mean=%.6f max=%.6f final=%.6f\n", samples, mean(errors),
           *std::max_element(errors.begin(), errors.end()), errors.back());
    printf("wrote %s/{summary.json,samples.csv,updates.csv}\n", out_dir.c_str());
    return deterministic ? 0 : 3;
} catch (const std::exception& error) {
    fprintf(stderr, "ERROR: %s\n", error.what());
    return 1;
}
