// Single-run tracking validation for the corrected-robot moving reference (examples/trajfiles/0_0).
// Runs ONE MPC tracking pass (one PCG exit tol) and reports mean/max/final EE tracking error plus the
// number of reference offsets reached, so config alignment (KNOT_POINTS / TIMESTEP / SIMULATION_PERIOD)
// can be A/B'd quickly without the 5-exit-tol paper sweep. Run from repo root (trajfile paths).
#include <fstream>
#include <vector>
#include <sstream>
#include <iostream>
#include <tuple>
#include <numeric>
#include <algorithm>
#include "mpcsim.cuh"
#include "dynamics/rbd_plant.cuh"
#include "settings.cuh"
#include "utils/experiment.cuh"
#include "gpu_pcg.cuh"

int main(int argc, char** argv){
    constexpr uint32_t state_size = grid::NUM_JOINTS*2;
    constexpr uint32_t control_size = grid::NUM_JOINTS;
    constexpr uint32_t knot_points = KNOT_POINTS;
    const linsys_t timestep = TIMESTEP;

    float pcg_exit_tol = (knot_points==32) ? 5e-6 : (knot_points==64 ? 5e-5 : 1e-5);

    std::string prefix = (argc > 1) ? argv[1] : "examples/trajfiles/0_0";
    std::vector<std::vector<linsys_t>> eePos_traj2d = readCSVToVecVec<linsys_t>((prefix+"_eepos.traj").c_str());
    std::vector<std::vector<linsys_t>> xu_traj2d    = readCSVToVecVec<linsys_t>((prefix+"_traj.csv").c_str());
    if(eePos_traj2d.size() < knot_points){ std::cout << "traj too short\n"; return 1; }

    std::vector<linsys_t> h_eePos_traj, h_xu_traj;
    for (const auto& v : eePos_traj2d) h_eePos_traj.insert(h_eePos_traj.end(), v.begin(), v.end());
    for (const auto& v : xu_traj2d)    h_xu_traj.insert(h_xu_traj.end(), v.begin(), v.end());

    linsys_t *d_eePos_traj, *d_xu_traj, *d_xs;
    gpuErrchk(cudaMalloc(&d_eePos_traj, h_eePos_traj.size()*sizeof(linsys_t)));
    gpuErrchk(cudaMemcpy(d_eePos_traj, h_eePos_traj.data(), h_eePos_traj.size()*sizeof(linsys_t), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc(&d_xu_traj, h_xu_traj.size()*sizeof(linsys_t)));
    gpuErrchk(cudaMemcpy(d_xu_traj, h_xu_traj.data(), h_xu_traj.size()*sizeof(linsys_t), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc(&d_xs, state_size*sizeof(linsys_t)));
    gpuErrchk(cudaMemcpy(d_xs, h_xu_traj.data(), state_size*sizeof(linsys_t), cudaMemcpyHostToDevice));

    printf("config: KNOT_POINTS=%d TIMESTEP=%g SIMULATION_PERIOD=%d CONST_UPDATE_FREQ=%d exit_tol=%g\n",
           KNOT_POINTS, (double)TIMESTEP, SIMULATION_PERIOD, CONST_UPDATE_FREQ, pcg_exit_tol);

    auto stats = simulateMPC<linsys_t, toplevel_return_type>(state_size, control_size, knot_points,
        static_cast<uint32_t>(eePos_traj2d.size()), timestep, d_eePos_traj, d_xu_traj, d_xs,
        0, 0, 0, pcg_exit_tol, std::string("tmp/results/validate"));

    std::vector<float> errs = std::get<1>(stats);
    float final_err = std::get<2>(stats);
    if (errs.empty()){ std::cout << "no tracking samples\n"; return 1; }
    float mean = std::accumulate(errs.begin(), errs.end(), 0.0f) / errs.size();
    float mx   = *std::max_element(errs.begin(), errs.end());
    printf("RESULT offsets=%zu mean=%.6f max=%.6f final=%.6f\n", errs.size(), mean, mx, final_err);
    // print a coarse trace so divergence is visible
    printf("trace: ");
    for (size_t i = 0; i < errs.size(); i += std::max((size_t)1, errs.size()/20)) printf("%.4f ", errs[i]);
    printf("\n");
    return 0;
}
