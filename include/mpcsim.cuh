#pragma once
#include <iomanip>
#include <fstream>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cstdint>
#include <cublas_v2.h>
#include <math.h>
#include <cmath>
#include <random>
#include <cuda_runtime.h>
#include <tuple>
#include <time.h>
#include "integrator.cuh"
#include "settings.cuh"
#include "utils/experiment.cuh"
#include "gpuassert.cuh"

#if LINSYS_SOLVE == 1
#include "pcg/sqp.cuh"
#else 
#include "qdldl/sqp.cuh"
#endif



template <typename T>
__global__
void compute_tracking_error_kernel(T *d_tracking_error, uint32_t state_size, T *d_xu_goal, T *d_xs){
    
    T err;
    
    for(int ind = threadIdx.x; ind < state_size/2; ind += blockDim.x){
        err = abs(d_xs[ind] - d_xu_goal[ind]);
        atomicAdd(d_tracking_error, err);
    }
}


template <typename T>
T compute_tracking_error(uint32_t state_size, T *d_xu_goal, T *d_xs){

    T h_tracking_error = 0.0f;
    T *d_tracking_error;
    gpuErrchk(cudaMalloc(&d_tracking_error, sizeof(T)));
    gpuErrchk(cudaMemcpy(d_tracking_error, &h_tracking_error, sizeof(T), cudaMemcpyHostToDevice));

    compute_tracking_error_kernel<T><<<1,32>>>(d_tracking_error, state_size, d_xu_goal, d_xs);

    gpuErrchk(cudaMemcpy(&h_tracking_error, d_tracking_error, sizeof(T), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(d_tracking_error));
    return h_tracking_error;
}


template <typename T>
void dump_tracking_data(std::vector<int> *pcg_iters, std::vector<bool> *pcg_exits, std::vector<double> *linsys_times, std::vector<double> *sqp_times, std::vector<uint32_t> *sqp_iters, 
                std::vector<bool> *sqp_exits, std::vector<T> *tracking_errors, std::vector<std::vector<T>> *tracking_path, uint32_t timesteps_taken, 
                uint32_t control_updates_taken, uint32_t start_state_ind, uint32_t goal_state_ind, uint32_t test_iter,
                std::string filename_prefix){
    // Helper function to create file names
    auto createFileName = [&](const std::string& data_type) {
        std::string filename = filename_prefix + "_" + std::to_string(test_iter) + "_" + data_type + ".result";
        return filename;
    };
    
    // Helper function to dump single-dimension vector data
    auto dumpVectorData = [&](const auto& data, const std::string& data_type) {
        std::ofstream file(createFileName(data_type));
        if (!file.is_open()) {
            std::cerr << "Failed to open " << data_type << " file.\n";
            return;
        }
        for (const auto& item : *data) {
            file << item << '\n';
        }
        file.close();
    };

    // Dump single-dimension vector data
    dumpVectorData(pcg_iters, "pcg_iters");
    dumpVectorData(linsys_times, "linsys_times");
    dumpVectorData(sqp_times, "sqp_times");
    dumpVectorData(sqp_iters, "sqp_iters");
    dumpVectorData(sqp_exits, "sqp_exits");
    dumpVectorData(tracking_errors, "tracking_errors");
    dumpVectorData(pcg_exits, "pcg_exits");


    // Dump two-dimension vector data (tracking_path)
    std::ofstream file(createFileName("tracking_path"));
    if (!file.is_open()) {
        std::cerr << "Failed to open tracking_path file.\n";
        return;
    }
    for (const auto& outerItem : *tracking_path) {
        for (const auto& innerItem : outerItem) {
            file << innerItem << ',';
        }
        file << '\n';
    }
    file.close();

    std::ofstream statsfile(createFileName("stats"));
    if (!statsfile.is_open()) {
        std::cerr << "Failed to open stats file.\n";
        return;
    }
    statsfile << "timesteps: " << timesteps_taken << "\n";
    statsfile << "control_updates: " << control_updates_taken << "\n";
    // printStatsToFile<double>(&linsys_times, )
    
    statsfile.close();
}


void print_test_config(){
    std::cout << "Knot points: " << KNOT_POINTS << "\n";
    std::cout << "State size: " << STATE_SIZE << "\n";
    std::cout << "Datatype: " << (USE_DOUBLES ? "DOUBLE" : "FLOAT") << "\n";
    std::cout << "Sqp exits condition: " << (CONST_UPDATE_FREQ ? "CONSTANT TIME" : "CONSTANT ITERS") << "\n";
    std::cout << "QD COST: " << QD_COST << "\n";
    std::cout << "R COST: " << R_COST << "\n";
    std::cout << "Rho factor: " << RHO_FACTOR << "\n";
    std::cout << "Rho max: " << RHO_MAX << "\n";
    std::cout << "Test iters: " << TEST_ITERS << "\n";
#if CONST_UPDATE_FREQ
    std::cout << "Max sqp time: " << SQP_MAX_TIME_US << "\n";
#else
    std::cout << "Max sqp iter: " << SQP_MAX_ITER << "\n";
#endif
    std::cout << "Solver: " << ( (LINSYS_SOLVE == 1) ? "PCG" : "QDLDL") << "\n";
#if LINSYS_SOLVE == 1
    std::cout << "Max pcg iter: " << PCG_MAX_ITER << "\n";
    // std::cout << "pcg exit tol: " << PCG_EXIT_TOL << "\n";
#endif
    std::cout << "Save data: " << (SAVE_DATA ? "ON" : "OFF") << "\n";
    std::cout << "Jitters: " << (REMOVE_JITTERS ? "ON" : "OFF") << "\n";

    std::cout << "\n\n";
}


template <typename T, typename return_type>
std::tuple<std::vector<toplevel_return_type>, std::vector<linsys_t>, linsys_t> simulateMPC(const uint32_t state_size, const uint32_t control_size, const uint32_t knot_points, const uint32_t traj_steps, 
            float timestep, T *d_eePos_traj, T *d_xu_traj, T *d_xs, uint32_t start_state_ind, uint32_t goal_state_ind, uint32_t test_iter, T linsys_exit_tol,
            std::string test_output_prefix){

    const uint32_t traj_len = (state_size+control_size)*knot_points-control_size;

    const T shift_threshold = SHIFT_THRESHOLD;
    const int max_control_updates = 100000;
    
    
    // struct timespec solve_start, solve_end;
    double sqp_solve_time_us = 0;               // current sqp solve time
    double simulation_time = 0;                 // current simulation time
    double prev_simulation_time = 0;            // last simulation time
    double time_since_timestep = 0;             // time since last timestep of original trajectory
    bool shifted = false;                       // has xu been shifted
    uint32_t traj_offset = 0;                        // current goal states of original trajectory


    // vars for recording data
    std::vector<std::vector<T>> tracking_path;      // list of traversed traj
    std::vector<int> linsys_iters;
    std::vector<double> linsys_times;
    std::vector<double> sqp_times;
    std::vector<uint32_t> sqp_iters;
    std::vector<bool> sqp_exits;
    std::vector<bool> linsys_exits;
    std::vector<T> tracking_errors;
    std::vector<T> joint_errors;   // JOINT_COST_MODE: |q_actual - q_ref| (positions) at each goal step
    std::vector<int> cur_linsys_iters;
    std::vector<bool> cur_linsys_exits;
    std::vector<double> cur_linsys_times;
    std::tuple<std::vector<int>, std::vector<double>, double, uint32_t, bool, std::vector<bool>> sqp_stats;
    uint32_t cur_sqp_iters;
    T cur_tracking_error;
    int control_update_step;


    // mpc iterates
    T *d_lambda, *d_eePos_goal, *d_xu, *d_xu_old;
    gpuErrchk(cudaMalloc(&d_lambda, state_size*knot_points*sizeof(T)));
    gpuErrchk(cudaMalloc(&d_xu, traj_len*sizeof(T)));
    gpuErrchk(cudaMalloc(&d_xu_old, traj_len*sizeof(T)));
    gpuErrchk(cudaMalloc(&d_eePos_goal, 6*knot_points*sizeof(T)));
    gpuErrchk(cudaMemset(d_lambda, 0, state_size*knot_points*sizeof(T)));
    gpuErrchk(cudaMemcpy(d_eePos_goal, d_eePos_traj, 6*knot_points*sizeof(T), cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemcpy(d_xu_old, d_xu_traj, traj_len*sizeof(T), cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemcpy(d_xu, d_xu_traj, traj_len*sizeof(T), cudaMemcpyDeviceToDevice));

    // Per-knot STATE goal for joint-space tracking. Extract the state-only reference from d_xu_traj
    // (xu-layout, stride state_size+control_size) into a contiguous (traj_steps*state_size) buffer,
    // and slide a knot_points window (d_xs_goal) exactly like d_eePos_goal. Passed to the solver only
    // in JOINT_COST_MODE; otherwise d_xs_goal stays nullptr => EE-only cost (the existing path).
    T *d_xs_goal = nullptr, *d_xs_goal_full = nullptr;
#if JOINT_COST_MODE
    gpuErrchk(cudaMalloc(&d_xs_goal,      state_size*knot_points*sizeof(T)));
    gpuErrchk(cudaMalloc(&d_xs_goal_full, state_size*traj_steps*sizeof(T)));
    gpuErrchk(cudaMemcpy2D(d_xs_goal_full, state_size*sizeof(T),
                           d_xu_traj, (state_size+control_size)*sizeof(T),
                           state_size*sizeof(T), traj_steps, cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemcpy(d_xs_goal, d_xs_goal_full, state_size*knot_points*sizeof(T), cudaMemcpyDeviceToDevice));
#endif


    void *d_dynmem = gato_plant::initializeDynamicsConstMem<T>();


    // temp host memory
    T h_xs[state_size];
    gpuErrchk(cudaMemcpy(h_xs, d_xs, state_size*sizeof(T), cudaMemcpyDeviceToHost));
    tracking_path.push_back(std::vector<T>(h_xs, &h_xs[state_size]));    
    gpuErrchk(cudaPeekAtLastError());
    T h_eePos[6];
    T h_eePos_goal[6];


    // temp device memory
    T *d_eePos;
    gpuErrchk(cudaMalloc(&d_eePos, 6*sizeof(T)));

#if LINSYS_SOLVE == 1
    pcg_config<T> config;
    config.pcg_block = PCG_NUM_THREADS;
    config.pcg_exit_tol = linsys_exit_tol;   // absolute floor on preconditioned residual eta
    config.pcg_rel_tol = PCG_RES_TOL;        // relative tol (eta vs eta_init), matches glass::pcg
    config.pcg_max_iter = PCG_MAX_ITER;
#endif

    T rho = RHO_INIT;
    T rho_reset = RHO_INIT;

#if REMOVE_JITTERS
	#if LINSYS_SOLVE == 1
#if PCG_TRUE_EXIT_CHECK_PERIOD
    // rel_tol is a TRUE-residual relative tol under the true-exit check (linear scale, not the
    // quadratic eta scale) — 1e-11 would be unreachable in float32 and the warm-up would spin
    // 100x10000 over-iterated solves (which measurably corrupts the warm start).
    config.pcg_exit_tol = 1e-8;
    config.pcg_rel_tol = 1e-4;    // tight warm-start solve (true-residual scale)
    config.pcg_max_iter = 2000;
#else
    config.pcg_exit_tol = 1e-11;
    config.pcg_rel_tol = 1e-11;   // tight warm-start solve
    config.pcg_max_iter = 10000;
#endif

    // Converge the initial trajectory on goal window 0 and KEEP it (mirrors GATO warm-starting XU and
    // keeping the solved trajectory). A good warm-start is essential: with SQP=1 real-time iteration the
    // tracking loop can only REFINE the trajectory, not build it, so starting cold from the raw
    // zero-control hold diverges on the stiff iiwa. (Previously each warm solve was discarded by resetting
    // d_xu to d_xu_traj — that left tracking starting cold.)
    for(int j = 0; j < 100; j++){
        sqpSolvePcg<T>(state_size, control_size, knot_points, timestep, d_eePos_goal, d_lambda, d_xu, d_dynmem, config, rho, RHO_INIT, d_xs_goal);
    }
    rho = RHO_INIT;
    config.pcg_exit_tol = linsys_exit_tol;
    config.pcg_rel_tol = PCG_RES_TOL;
    config.pcg_max_iter = PCG_MAX_ITER;
	#else
    for(int j = 0; j < 100; j++){
        sqpSolveQdldl<T>(state_size, control_size, knot_points, timestep, d_eePos_goal, d_lambda, d_xu, d_dynmem, rho, RHO_INIT, d_xs_goal);
    }
    rho = RHO_INIT;
	#endif

#endif // #if REMOVE_JITTERS



    //
    // MPC tracking loop
    //
    for(control_update_step = 0; control_update_step < max_control_updates; control_update_step++){
        

        if (traj_offset == traj_steps){ break; }



#if LIVE_PRINT_PATH
        grid::end_effector_pose_kernel_EE<T><<<1,128,grid::END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(d_eePos, d_xs, grid::NUM_JOINTS, (grid::robotModel<T> *) d_dynmem, 1);
        gpuErrchk(cudaMemcpy(h_eePos, d_eePos, 6*sizeof(T), cudaMemcpyDeviceToHost));
        for (uint32_t i = 0; i < 6; i++){
            std::cout << h_eePos[i] << (i < 5 ? " " : "\n");
        }
#endif // #if LIVE_PRINT_PATH
        


#if LINSYS_SOLVE == 1
        sqp_stats = sqpSolvePcg<T>(state_size, control_size, knot_points, timestep, d_eePos_goal, d_lambda, d_xu, d_dynmem, config, rho, rho_reset, d_xs_goal);
#else 
	    sqp_stats = sqpSolveQdldl<T>(state_size, control_size, knot_points, timestep, d_eePos_goal, d_lambda, d_xu, d_dynmem, rho, rho_reset, d_xs_goal);
#endif

        cur_linsys_iters = std::get<0>(sqp_stats);
        cur_linsys_times = std::get<1>(sqp_stats);
        sqp_solve_time_us = std::get<2>(sqp_stats);
        cur_sqp_iters = std::get<3>(sqp_stats);
        sqp_exits.push_back(std::get<4>(sqp_stats));
        cur_linsys_exits = std::get<5>(sqp_stats);


#if CONST_UPDATE_FREQ
        simulation_time = SIMULATION_PERIOD;
#else
        simulation_time = sqp_solve_time_us;
#endif
        

        // simulate traj for current solve time, offset by previous solve time
#ifdef APPLY_FRESH_CONTROL
        // experiment: apply the CURRENT solve's controls, aligned to the position within the
        // knot window (mirrors GATO's apply-fresh-control-immediately loop; the default path
        // applies the PREVIOUS solve's trajectory at a fixed one-period offset).
        simple_simulate<T>(state_size, control_size, knot_points, d_xs, d_xu, d_dynmem, timestep, time_since_timestep*1e6, simulation_time);
#else
        simple_simulate<T>(state_size, control_size, knot_points, d_xs, d_xu_old, d_dynmem, timestep, prev_simulation_time, simulation_time);
#endif

        // old xu = new xu
        gpuErrchk(cudaMemcpy(d_xu_old, d_xu, traj_len*sizeof(T), cudaMemcpyDeviceToDevice));


        time_since_timestep += simulation_time * 1e-6;

        // if shift_threshold% through timestep
        if (!shifted && time_since_timestep > shift_threshold){
            
            // record tracking error
            grid::end_effector_pose_kernel_EE<T><<<1,128,grid::END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(d_eePos, d_xs, grid::NUM_JOINTS, (grid::robotModel<T> *) d_dynmem, 1);
            gpuErrchk(cudaMemcpy(h_eePos, d_eePos, 6*sizeof(T), cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpy(h_eePos_goal, d_eePos_goal, 6*sizeof(T), cudaMemcpyDeviceToHost));
            // L2 position error — MUST match the GATO/BatchThneed harnesses (np.linalg.norm)
            // for the 3-way comparison; the old L1 sum inflated MPCGPU's numbers 1.3-1.7x.
            cur_tracking_error = 0.0;
            for(uint32_t i=0; i < 3; i++){
                T d = h_eePos[i] - h_eePos_goal[i];
                cur_tracking_error += d*d;
            }
            cur_tracking_error = sqrt(cur_tracking_error);
            // std::cout << cur_tracking_error << std::endl;;
            tracking_errors.push_back(cur_tracking_error);
#if JOINT_COST_MODE
            // joint-space tracking error: |q_actual - q_ref| over the NQ positions, vs the reference
            // state at the current goal index (d_xs_goal_full holds the per-step q_ref/qd_ref).
            {
                T h_q[64]; T h_qref[64];
                const uint32_t nq = state_size/2;
                gpuErrchk(cudaMemcpy(h_q, d_xs, nq*sizeof(T), cudaMemcpyDeviceToHost));
                gpuErrchk(cudaMemcpy(h_qref, &d_xs_goal_full[traj_offset*state_size], nq*sizeof(T), cudaMemcpyDeviceToHost));
                T je = 0; for(uint32_t i=0;i<nq;i++) je += abs(h_q[i]-h_qref[i]);
                joint_errors.push_back(je);
            }
#endif

            traj_offset++;

            // shift xu — warm-start = time-shift + DUPLICATE the last stage (real-time-iteration),
            // matching GATO and the CPU baseline for the fair 3-way comparison. just_shift moves knots
            // 0..N-2 <- 1..N-1 and leaves the old last stage [u_{N-2}, x_{N-1}] in place, which IS the
            // duplicated tail. (Previously the tail was refilled from the reference d_xu_traj; with a
            // static-hold reference that re-seeded "hold" every step and prevented cost-driven tracking.)
            just_shift<T>(state_size, control_size, knot_points, d_xu);
#ifdef CONSISTENT_SHIFT_TAIL
            // Overwrite the duplicated tail state with a dynamically-consistent rollout
            // x_{N-1} = f(x_{N-2}, u_{N-2}) under the SOLVER's integrator, so the warm start
            // carries no artificial defect at the last constraint row (the dup-last-stage
            // shift concentrates the ENTIRE warm-start defect there, and an exact QP solve
            // spends its step closing that artifact instead of tracking).
            {
                const uint32_t ssc = state_size + control_size;
                gpuErrchk(cudaMemcpy(&d_xu[(knot_points-1)*ssc], &d_xu[(knot_points-2)*ssc], state_size*sizeof(T), cudaMemcpyDeviceToDevice));
                const size_t tail_smem = sizeof(T)*(2*state_size + control_size + state_size/2 + gato_plant::forwardDynamicsAndGradient_TempMemSize_Shared());
                simple_integrator_kernel<T><<<1,32,tail_smem>>>(state_size, control_size, &d_xu[(knot_points-1)*ssc], &d_xu[(knot_points-2)*ssc+state_size], d_dynmem, timestep);
            }
#endif

            // shift goal
            just_shift(6, 0, knot_points, d_eePos_goal);
            if (traj_offset + knot_points < traj_steps){
                gpuErrchk(cudaMemcpy(&d_eePos_goal[(knot_points-1)*(6)], &d_eePos_traj[(traj_offset+knot_points-1) * (6)], 6*sizeof(T), cudaMemcpyDeviceToDevice));
            }
            else{
                // fill in last goal state with goal state and zero velocity
                gpuErrchk(cudaMemcpy(&d_eePos_goal[(knot_points-1)*(6)], &d_eePos_traj[(traj_steps-1)*(6)], (6)*sizeof(T), cudaMemcpyDeviceToDevice));
                // gpuErrchk(cudaMemset(&d_eePos_goal[(knot_points-1)*(6) + state_size / 2], 0, (state_size/2) * sizeof(T)));
            }
#if JOINT_COST_MODE
            // shift the per-knot state goal in lockstep with d_eePos_goal
            just_shift<T>(state_size, 0, knot_points, d_xs_goal);
            {
                uint32_t src = (traj_offset + knot_points < traj_steps) ? (traj_offset+knot_points-1) : (traj_steps-1);
                gpuErrchk(cudaMemcpy(&d_xs_goal[(knot_points-1)*state_size], &d_xs_goal_full[src*state_size], state_size*sizeof(T), cudaMemcpyDeviceToDevice));
            }
#endif
            
            // shift lambda
            just_shift(state_size, 0, knot_points, d_lambda);
                // gpuErrchk(cudaMemset(&lambdas[i][state_size*(knot_points-1)], 0, state_size*sizeof(T)));
            
            shifted = true;
        }

        if (time_since_timestep > timestep){
            // std::cout << "shifted to offset: " << traj_offset + 1 << std::endl;
            shifted = false;
            time_since_timestep = std::fmod(time_since_timestep, timestep);
        }
        gpuErrchk(cudaMemcpy(d_xu, d_xs, state_size*sizeof(T), cudaMemcpyDeviceToDevice));


        
        prev_simulation_time = simulation_time;

        gpuErrchk(cudaPeekAtLastError());

        
        // record data
        linsys_iters.insert(linsys_iters.end(), cur_linsys_iters.begin(), cur_linsys_iters.end());                      // linsys iters
        linsys_times.insert(linsys_times.end(), cur_linsys_times.begin(), cur_linsys_times.end());          // linsys times
        linsys_exits.insert(linsys_exits.end(), cur_linsys_exits.begin(), cur_linsys_exits.end());
        gpuErrchk(cudaMemcpy(h_xs, d_xs, state_size*sizeof(T), cudaMemcpyDeviceToHost));
        tracking_path.push_back(std::vector<T>(h_xs, &h_xs[state_size]));                                   // next state
        sqp_times.push_back(sqp_solve_time_us);
        sqp_iters.push_back(cur_sqp_iters);


#if LIVE_PRINT_STATS
        if (control_update_step % 1000 == 50){
            for (uint32_t i = 0; i < state_size; i++){
                std::cout << h_xs[i] << (i < state_size-1 ? " " : "\n");
            }
    #if TIME_LINSYS == 1
            std::cout << "linear system solve time:" << std::endl;
            printStats<double>(&linsys_times);
    #endif // #if TIME_LINSYS
            std::cout << "goal offset [" << traj_offset << "]\n";
            std::cout << "sqp iters" << std::endl;
            printStats<uint32_t>(&sqp_iters);
            std::cout << "sqp times" << std::endl;
            printStats<double>(&sqp_times);
            
            int totalOnes = std::accumulate(linsys_exits.begin(), linsys_exits.end(), 0);
            double max_iter_pct = (static_cast<double>(totalOnes) / linsys_exits.size());
            std::cout << "linsys exits for max iter: " << max_iter_pct * 100 << "% of the time\n";
            if (max_iter_pct > 0.5) {
               std::cout << "WARNING: PCG exiting for max iter over 50% of the time" << std::endl;
            }
            
            std::cout << "avg tracking error: " << std::accumulate(tracking_errors.begin(), tracking_errors.end(), 0.0f) / traj_offset << " current error: " << cur_tracking_error << "\n";
            std::cout << std::endl;

        }

#endif


    }
#if SAVE_DATA
    dump_tracking_data(&linsys_iters, &linsys_exits, &linsys_times, &sqp_times, &sqp_iters, &sqp_exits, &tracking_errors, &tracking_path,
            traj_offset, control_update_step, start_state_ind, goal_state_ind, test_iter, test_output_prefix);
#endif
#ifdef PRINT_LINSYS_ITERS
    if(!linsys_iters.empty()){
        long sum = 0; int mn = linsys_iters[0], mx = linsys_iters[0];
        for(int v : linsys_iters){ sum += v; mn = std::min(mn,v); mx = std::max(mx,v); }
        long capped = std::accumulate(linsys_exits.begin(), linsys_exits.end(), 0L);
        printf("PCG_ITERS solves=%zu avg=%.1f min=%d max=%d  maxiter_exit=%.1f%%\n",
               linsys_iters.size(), (double)sum/linsys_iters.size(), mn, mx,
               100.0*capped/linsys_exits.size());
    }
#endif
#if JOINT_COST_MODE
    if(!joint_errors.empty()){
        double js = 0; T jmx = joint_errors[0];
        for(T v : joint_errors){ js += v; if(v>jmx) jmx=v; }
        printf("JOINT_ERR steps=%zu mean=%.6f max=%.6f final=%.6f  (sum|q_actual-q_ref| over %d joints)\n",
               joint_errors.size(), js/joint_errors.size(), (double)jmx, (double)joint_errors.back(), state_size/2);
    }
#endif
    

    grid::end_effector_pose_kernel_EE<T><<<1,128,grid::END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(d_eePos, d_xs, grid::NUM_JOINTS, (grid::robotModel<T> *) d_dynmem, 1);
    gpuErrchk(cudaMemcpy(h_eePos, d_eePos, 6*sizeof(T), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_eePos_goal, d_eePos_goal, 6*sizeof(T), cudaMemcpyDeviceToHost));
    cur_tracking_error = 0.0;
    for(uint32_t i=0; i < 3; i++){
        cur_tracking_error += abs(h_eePos[i] - h_eePos_goal[i]);
    }

    gato_plant::freeDynamicsConstMem<T>(d_dynmem);

    gpuErrchk(cudaFree(d_lambda));
    gpuErrchk(cudaFree(d_xu));
    gpuErrchk(cudaFree(d_eePos_goal));
#if JOINT_COST_MODE
    gpuErrchk(cudaFree(d_xs_goal));
    gpuErrchk(cudaFree(d_xs_goal_full));
#endif
    gpuErrchk(cudaFree(d_xu_old));

    gpuErrchk(cudaFree(d_eePos));

    #if TIME_LINSYS == 1 
        return std::make_tuple(linsys_times, tracking_errors, cur_tracking_error);
    #else
        return std::make_tuple(sqp_iters, tracking_errors, cur_tracking_error);
    #endif
}
