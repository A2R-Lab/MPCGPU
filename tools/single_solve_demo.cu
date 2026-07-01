// Open-loop SINGLE-SOLVE demo / validation (sidesteps the closed-loop stability issue). Loads the shared
// reference, perturbs x0 off the reference, and runs ONE MPC solve (sqpSolvePcg). Reports the PCG iteration
// count and how far the solved knot-1 state moved from the perturbed x0 toward the reference. Built twice:
//   JOINT mode (JOINT_COST_MODE=1, Q_COST>0, EE_COST=0): full-rank state Hessian => well-conditioned Schur
//   EE    mode (JOINT_COST_MODE=0, EE_COST>0, Q_COST=0): EE-position-only => rank-deficient => ill-conditioned
// Expectation (the conditioning conclusion): joint-mode PCG converges in far fewer iters than EE-mode.
// Run from repo root. Build: see tools comment / Makefile-style nvcc line in the session notes.
#include <fstream>
#include <vector>
#include <sstream>
#include <iostream>
#include <numeric>
#include <cmath>
#include "mpcsim.cuh"
#include "dynamics/rbd_plant.cuh"
#include "settings.cuh"
#include "utils/experiment.cuh"
#include "gpu_pcg.cuh"

int main(int argc, char** argv){
    const uint32_t state_size = grid::NUM_JOINTS*2, control_size = grid::NUM_JOINTS, knot_points = KNOT_POINTS;
    const uint32_t nq = grid::NUM_JOINTS;
    const linsys_t timestep = TIMESTEP;
    const uint32_t traj_len = (state_size+control_size)*knot_points-control_size;
    std::string prefix = (argc > 1) ? argv[1] : "examples/trajfiles/0_0";
    float perturb = (argc > 2) ? atof(argv[2]) : 0.15f;   // rad added to each joint of x0

    auto eePos2d = readCSVToVecVec<linsys_t>((prefix+"_eepos.traj").c_str());
    auto xu2d    = readCSVToVecVec<linsys_t>((prefix+"_traj.csv").c_str());
    if(eePos2d.size() < knot_points){ std::cout << "traj too short\n"; return 1; }

    // flatten reference window [0, knot_points)
    std::vector<linsys_t> h_ee, h_xu_ref;
    for(uint32_t k=0;k<knot_points;k++) for(auto v: eePos2d[k]) h_ee.push_back(v);
    for(uint32_t k=0;k<knot_points;k++){
        for(uint32_t i=0;i<state_size;i++) h_xu_ref.push_back(xu2d[k][i]);
        if(k<knot_points-1) for(uint32_t i=0;i<control_size;i++) h_xu_ref.push_back(xu2d[k][state_size+i]);
    }
    // state-only goal window (NX/knot)
    std::vector<linsys_t> h_xsgoal;
    for(uint32_t k=0;k<knot_points;k++) for(uint32_t i=0;i<state_size;i++) h_xsgoal.push_back(xu2d[k][i]);

    // perturbed x0 = reference state 0 + perturb on each joint position
    std::vector<linsys_t> h_xs(state_size);
    for(uint32_t i=0;i<state_size;i++) h_xs[i] = xu2d[0][i];
    for(uint32_t i=0;i<nq;i++) h_xs[i] += perturb;

    linsys_t *d_eePos,*d_xu,*d_lambda,*d_xs,*d_xsgoal=nullptr;
    gpuErrchk(cudaMalloc(&d_eePos, h_ee.size()*sizeof(linsys_t)));
    gpuErrchk(cudaMalloc(&d_xu, traj_len*sizeof(linsys_t)));
    gpuErrchk(cudaMalloc(&d_lambda, state_size*knot_points*sizeof(linsys_t)));
    gpuErrchk(cudaMalloc(&d_xs, state_size*sizeof(linsys_t)));
    gpuErrchk(cudaMemcpy(d_eePos, h_ee.data(), h_ee.size()*sizeof(linsys_t), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_xu, h_xu_ref.data(), traj_len*sizeof(linsys_t), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemset(d_lambda, 0, state_size*knot_points*sizeof(linsys_t)));
    gpuErrchk(cudaMemcpy(d_xs, h_xs.data(), state_size*sizeof(linsys_t), cudaMemcpyHostToDevice));
#if JOINT_COST_MODE
    gpuErrchk(cudaMalloc(&d_xsgoal, h_xsgoal.size()*sizeof(linsys_t)));
    gpuErrchk(cudaMemcpy(d_xsgoal, h_xsgoal.data(), h_xsgoal.size()*sizeof(linsys_t), cudaMemcpyHostToDevice));
#endif

    void *d_dynmem = gato_plant::initializeDynamicsConstMem<linsys_t>();
    pcg_config<linsys_t> config;
    config.pcg_block = PCG_NUM_THREADS;
    config.pcg_exit_tol = 1e-6; config.pcg_rel_tol = PCG_RES_TOL; config.pcg_max_iter = PCG_MAX_ITER;
    linsys_t rho = RHO_INIT, rho_reset = RHO_INIT;

    auto stats = sqpSolvePcg<linsys_t>(state_size, control_size, knot_points, timestep,
                  d_eePos, d_lambda, d_xu, d_dynmem, config, rho, rho_reset, d_xsgoal);

    std::vector<int> li = std::get<0>(stats);
    uint32_t sqp_iters = std::get<3>(stats);
    double avg = li.empty()?0: (double)std::accumulate(li.begin(),li.end(),0)/li.size();
    int mx = li.empty()?0:*std::max_element(li.begin(),li.end());

    // read back solved knot-1 state, measure how far it moved from perturbed x0 toward the reference q_ref[1]
    std::vector<linsys_t> h_sol(traj_len);
    gpuErrchk(cudaMemcpy(h_sol.data(), d_xu, traj_len*sizeof(linsys_t), cudaMemcpyDeviceToHost));
    const uint32_t k1 = state_size+control_size;  // offset of knot 1 state
    double d_ref=0, d_start=0;
    for(uint32_t i=0;i<nq;i++){
        d_ref   += std::fabs(h_sol[k1+i] - xu2d[1][i]);     // solved knot1 q  vs reference q_ref[1]
        d_start += std::fabs(h_sol[k1+i] - h_xs[i]);        // solved knot1 q  vs perturbed start
    }
    printf("%s  perturb=%.3f  SQP_iters=%u  PCG_iters avg=%.1f max=%d (cap=%d)  |q1-q_ref|=%.4f |q1-q0pert|=%.4f\n",
           JOINT_COST_MODE? "JOINT":"EE  ", perturb, sqp_iters, avg, mx, PCG_MAX_ITER, d_ref, d_start);
    return 0;
}
