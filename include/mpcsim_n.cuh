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
#include "mpcsim.cuh"

#if LINSYS_SOLVE == 1
#include "pcg/sqp.cuh"
#else 
#include "qdldl/sqp.cuh"
#endif



template <typename T, typename return_type>
std::tuple<std::vector<toplevel_return_type>, std::vector<linsys_t>, linsys_t> simulateMPC_n(
    const uint32_t state_size, // constant
    const uint32_t control_size, // constant
    const uint32_t knot_points, // probably constant?
    const uint32_t traj_steps, // probably constant?
    float timestep, // constant
    T *d_eePos_traj, // eventually instanced
    T *d_xu_traj, // eventually instanced
    T *d_xs, // eventually instanced
    uint32_t start_state_ind, // eventually instanced
    uint32_t goal_state_ind, // eventually instanced
    uint32_t test_iter, // probably eventually instanced?
    T linsys_exit_tol, // probably constant
    std::string test_output_prefix, // probably constant
    uint32_t solve_count // constant
){

    // constant
    const uint32_t traj_len = (state_size+control_size)*knot_points-control_size;

    // constants
    const T shift_threshold = SHIFT_THRESHOLD;
    const int max_control_updates = 100000;
    
    
    // struct timespec solve_start, solve_end;
    std::vector<double> sqp_solve_time_us_vec(solve_count, 0);               // current sqp solve time
    std::vector<double> simulation_time_vec(solve_count, 0);                 // current simulation time
    std::vector<double> prev_simulation_time_vec(solve_count, 0);            // last simulation time
    std::vector<double> time_since_timestep_vec(solve_count, 0);             // time since last timestep of original trajectory
    std::vector<bool> shifted_vec(solve_count, false);                       // has xu been shifted
    std::vector<uint32_t> traj_offset_vec(solve_count, 0);                        // current goal states of original trajectory


    // vars for recording data
    std::vector<std::vector<std::vector<T>>> tracking_path_vec(solve_count);      // list of traversed traj
    std::vector<std::vector<int>> linsys_iters_vec(solve_count);
    std::vector<std::vector<double>> linsys_times_vec(solve_count);
    std::vector<std::vector<double>> sqp_times_vec(solve_count);
    std::vector<std::vector<uint32_t>> sqp_iters_vec(solve_count);
    std::vector<std::vector<bool>> sqp_exits_vec(solve_count);
    std::vector<std::vector<bool>> linsys_exits_vec(solve_count);
    std::vector<std::vector<T>> tracking_errors_vec(solve_count);
    std::vector<std::vector<int>> cur_linsys_iters_vec(solve_count);
    std::vector<std::vector<bool>> cur_linsys_exits_vec(solve_count);
    std::vector<std::vector<double>> cur_linsys_times_vec(solve_count);
    std::vector<std::tuple<std::vector<int>, std::vector<double>, double, uint32_t, bool, std::vector<bool>>> sqp_stats_vec(solve_count);
    std::vector<uint32_t> cur_sqp_iters_vec(solve_count);
    std::vector<T> cur_tracking_error_vec(solve_count);
    std::vector<int> control_update_step_vec(solve_count);


    // mpc iterates
    // TODO probably need to be instanced: one per simultaneous solve
    T *d_lambda, *d_eePos_goal, *d_xu, *d_xu_old;
    auto lambda_size = state_size*knot_points*sizeof(T)*solve_count;
    auto xu_size = traj_len * sizeof(T) * solve_count;
    auto eePos_size = 6*knot_points*sizeof(T) * solve_count;

    gpuErrchk(cudaMalloc(&d_lambda, lambda_size));
    gpuErrchk(cudaMalloc(&d_xu, xu_size));
    gpuErrchk(cudaMalloc(&d_xu_old, xu_size));
    gpuErrchk(cudaMalloc(&d_eePos_goal, eePos_size));
    gpuErrchk(cudaMemset(d_lambda, 0, lambda_size));
    gpuErrchk(cudaMemcpy(d_eePos_goal, d_eePos_traj, eePos_size, cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemcpy(d_xu_old, d_xu_traj, xu_size, cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemcpy(d_xu, d_xu_traj, xu_size, cudaMemcpyDeviceToDevice));


    // puts constants onto device, probably no need to instance
    void *d_dynmem = gato_plant::initializeDynamicsConstMem<T>();


    // temp host memory
    // TODO probably instanced
    auto xs_size = state_size * solve_count;
    T h_xs[xs_size];
    // this size needs to match the parameter passed to simulateMPC
    gpuErrchk(cudaMemcpy(h_xs, d_xs, state_size * sizeof(T), cudaMemcpyDeviceToHost));
    // this line uses pointers as iterators, essentially h_xs.begin(), h_xs.end()
    // doesn't need to be changed while we just pass in the first entry
    // but needs to be in a loop later
    tracking_path_vec[0].push_back(std::vector<T>(h_xs, &h_xs[state_size]));
    gpuErrchk(cudaPeekAtLastError());
    T h_eePos[6 * solve_count];
    T h_eePos_goal[6 * solve_count];


    // temp device memory
    // TODO probably instanced
    T *d_eePos;
    gpuErrchk(cudaMalloc(&d_eePos, sizeof(h_eePos)));

#if LINSYS_SOLVE == 1
    pcg_config<T> config;
    config.pcg_block = PCG_NUM_THREADS;
    config.pcg_exit_tol = linsys_exit_tol;
    config.pcg_max_iter = PCG_MAX_ITER;
#endif

    T rho = 1e-3;
    T rho_reset = 1e-3;

// TODO can probably ignore
#if REMOVE_JITTERS
	#if LINSYS_SOLVE == 1
    config.pcg_exit_tol = 1e-11;
    config.pcg_max_iter = 10000;
    
    for(int j = 0; j < 100; j++){
        sqpSolvePcg<T>(state_size, control_size, knot_points, timestep, d_eePos_goal, d_lambda, d_xu, d_dynmem, config, rho, 1e-3);
        gpuErrchk(cudaMemcpy(d_xu, d_xu_traj, traj_len*sizeof(T), cudaMemcpyDeviceToDevice));
    }
    rho = 1e-3;
    config.pcg_exit_tol = linsys_exit_tol;
    config.pcg_max_iter = PCG_MAX_ITER;
	#else
    for(int j = 0; j < 100; j++){
        sqpSolveQdldl<T>(state_size, control_size, knot_points, timestep, d_eePos_goal, d_lambda, d_xu, d_dynmem, rho, 1e-3);
        gpuErrchk(cudaMemcpy(d_xu, d_xu_traj, traj_len*sizeof(T), cudaMemcpyDeviceToDevice));
    }
    rho = 1e-3;
	#endif

#endif // #if REMOVE_JITTERS



    //
    // MPC tracking loop
    //
    for(control_update_step_vec[0] = 0; control_update_step_vec[0] < max_control_updates; control_update_step_vec[0]++){
        

        if (traj_offset_vec[0] == traj_steps){ break; }


// don't need to change this for now, it's just doing the 0th problem but
// TODO update this to print all $solve_count paths
#if LIVE_PRINT_PATH
        grid::end_effector_positions_kernel<T><<<1,128>>>(d_eePos, d_xs, grid::NUM_JOINTS, (grid::robotModel<T> *) d_dynmem, 1);
        gpuErrchk(cudaMemcpy(h_eePos, d_eePos, 6*sizeof(T), cudaMemcpyDeviceToHost));
        for (uint32_t i = 0; i < 6; i++){
            std::cout << h_eePos[i] << (i < 5 ? " " : "\n");
        }
#endif // #if LIVE_PRINT_PATH
        


#if LINSYS_SOLVE == 1
        sqp_stats_vec[0] = sqpSolvePcg<T>(state_size, control_size, knot_points, timestep, d_eePos_goal, d_lambda, d_xu, d_dynmem, config, rho, rho_reset);
#else 
	    sqp_stats_vec[0] = sqpSolveQdldl<T>(state_size, control_size, knot_points, timestep, d_eePos_goal, d_lambda, d_xu, d_dynmem, rho, rho_reset);
#endif

        cur_linsys_iters_vec[0] = std::get<0>(sqp_stats_vec[0]);
        cur_linsys_times_vec[0] = std::get<1>(sqp_stats_vec[0]);
        sqp_solve_time_us_vec[0] = std::get<2>(sqp_stats_vec[0]);
        cur_sqp_iters_vec[0] = std::get<3>(sqp_stats_vec[0]);
        sqp_exits_vec[0].push_back(std::get<4>(sqp_stats_vec[0]));
        cur_linsys_exits_vec[0] = std::get<5>(sqp_stats_vec[0]);


#if CONST_UPDATE_FREQ
        simulation_time_vec[0] = SIMULATION_PERIOD;
#else
        simulation_time_vec[0] = sqp_solve_time_us_vec[0];
#endif
        

        // simulate traj for current solve time, offset by previous solve time
        simple_simulate<T>(state_size, control_size, knot_points, d_xs, d_xu_old, d_dynmem, timestep, prev_simulation_time_vec[0], simulation_time_vec[0]);

        // old xu = new xu
        gpuErrchk(cudaMemcpy(d_xu_old, d_xu, traj_len*sizeof(T), cudaMemcpyDeviceToDevice));


        time_since_timestep_vec[0] += simulation_time_vec[0] * 1e-6;

        // if shift_threshold% through timestep
        if (!shifted_vec[0] && time_since_timestep_vec[0] > shift_threshold){
            
            // record tracking error
            grid::end_effector_positions_kernel<T><<<1,128>>>(d_eePos, d_xs, grid::NUM_JOINTS, (grid::robotModel<T> *) d_dynmem, 1);
            gpuErrchk(cudaMemcpy(h_eePos, d_eePos, 6*sizeof(T), cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpy(h_eePos_goal, d_eePos_goal, 6*sizeof(T), cudaMemcpyDeviceToHost));
            cur_tracking_error_vec[0] = 0.0;
            for(uint32_t i=0; i < 3; i++){
                cur_tracking_error_vec[0] += abs(h_eePos[i] - h_eePos_goal[i]);
            }
            // std::cout << cur_tracking_error_vec[0] << std::endl;;
            tracking_errors_vec[0].push_back(cur_tracking_error_vec[0]);
            
            traj_offset_vec[0]++;

            // shift xu
            just_shift<T>(state_size, control_size, knot_points, d_xu);             // shift everything over one
            if (traj_offset_vec[0] + knot_points < traj_steps){
                // if within precomputed traj, fill in last state, control with precompute
                gpuErrchk(cudaMemcpy(&d_xu[traj_len - (state_size + control_size)], &d_xu_traj[(state_size+control_size)*traj_offset_vec[0] - control_size], sizeof(T)*(state_size+control_size), cudaMemcpyDeviceToDevice));     // last state filled from precomputed trajectory
            }
            else{
                // fill in last state with goal position, zero velocity, last control with zero control
                gpuErrchk(cudaMemcpy(&d_xu[traj_len - state_size], &d_xu_traj[(traj_steps-1)*(state_size+control_size)], (state_size/2)*sizeof(T), cudaMemcpyDeviceToDevice));
                gpuErrchk(cudaMemset(&d_xu[traj_len - state_size / 2], 0, (state_size/2) * sizeof(T)));
                gpuErrchk(cudaMemset(&d_xu[traj_len - (state_size+control_size)], 0, control_size * sizeof(T)));
            }
            
            // shift goal
            just_shift(6, 0, knot_points, d_eePos_goal);
            if (traj_offset_vec[0] + knot_points < traj_steps){
                gpuErrchk(cudaMemcpy(&d_eePos_goal[(knot_points-1)*(6)], &d_eePos_traj[(traj_offset_vec[0]+knot_points-1) * (6)], 6*sizeof(T), cudaMemcpyDeviceToDevice));
            }
            else{
                // fill in last goal state with goal state and zero velocity
                gpuErrchk(cudaMemcpy(&d_eePos_goal[(knot_points-1)*(6)], &d_eePos_traj[(traj_steps-1)*(6)], (6)*sizeof(T), cudaMemcpyDeviceToDevice));
                // gpuErrchk(cudaMemset(&d_eePos_goal[(knot_points-1)*(6) + state_size / 2], 0, (state_size/2) * sizeof(T)));
            }
            
            // shift lambda
            just_shift(state_size, 0, knot_points, d_lambda);
                // gpuErrchk(cudaMemset(&lambdas[i][state_size*(knot_points-1)], 0, state_size*sizeof(T)));
            
            shifted_vec[0] = true;
        }

        if (time_since_timestep_vec[0] > timestep){
            // std::cout << "shifted to offset: " << traj_offset_vec[0] + 1 << std::endl;
            shifted_vec[0] = false;
            time_since_timestep_vec[0] = std::fmod(time_since_timestep_vec[0], timestep);
        }
        gpuErrchk(cudaMemcpy(d_xu, d_xs, state_size*sizeof(T), cudaMemcpyDeviceToDevice));


        
        prev_simulation_time_vec[0] = simulation_time_vec[0];

        gpuErrchk(cudaPeekAtLastError());

        
        // record data
        linsys_iters_vec[0].insert(linsys_iters_vec[0].end(), cur_linsys_iters_vec[0].begin(), cur_linsys_iters_vec[0].end());                      // linsys iters
        linsys_times_vec[0].insert(linsys_times_vec[0].end(), cur_linsys_times_vec[0].begin(), cur_linsys_times_vec[0].end());          // linsys times
        linsys_exits_vec[0].insert(linsys_exits_vec[0].end(), cur_linsys_exits_vec[0].begin(), cur_linsys_exits_vec[0].end());
        gpuErrchk(cudaMemcpy(h_xs, d_xs, state_size*sizeof(T), cudaMemcpyDeviceToHost));
        tracking_path_vec[0].push_back(std::vector<T>(h_xs, &h_xs[state_size]));                                   // next state
        sqp_times_vec[0].push_back(sqp_solve_time_us_vec[0]);
        sqp_iters_vec[0].push_back(cur_sqp_iters_vec[0]);


#if LIVE_PRINT_STATS
        if (control_update_step_vec[0] % 1000 == 50){
            for (uint32_t i = 0; i < state_size; i++){
                std::cout << h_xs[i] << (i < state_size-1 ? " " : "\n");
            }
    #if TIME_LINSYS == 1
            std::cout << "linear system solve time:" << std::endl;
            printStats<double>(&linsys_times_vec[0]);
    #endif // #if TIME_LINSYS
            std::cout << "goal offset [" << traj_offset_vec[0] << "]\n";
            std::cout << "sqp iters" << std::endl;
            printStats<uint32_t>(&sqp_iters_vec[0]);
            std::cout << "sqp times" << std::endl;
            printStats<double>(&sqp_times_vec[0]);
            
            int totalOnes = std::accumulate(linsys_exits_vec[0].begin(), linsys_exits_vec[0].end(), 0);
            double max_iter_pct = (static_cast<double>(totalOnes) / linsys_exits_vec[0].size());
            std::cout << "linsys exits for max iter: " << max_iter_pct * 100 << "% of the time\n";
            if (max_iter_pct > 0.5) {
               std::cout << "WARNING: PCG exiting for max iter over 50% of the time" << std::endl;
            }
            
            std::cout << "avg tracking error: " << std::accumulate(tracking_errors_vec[0].begin(), tracking_errors_vec[0].end(), 0.0f) / traj_offset_vec[0] << " current error: " << cur_tracking_error_vec[0] << "\n";
            std::cout << std::endl;

        }

#endif


    }
#if SAVE_DATA
    dump_tracking_data(&linsys_iters_vec[0], &linsys_exits_vec[0], &linsys_times_vec[0], &sqp_times_vec[0], &sqp_iters_vec[0], &sqp_exits, &tracking_errors_vec[0], &tracking_path_vec[0], 
            traj_offset_vec[0], control_update_step_vec[0], start_state_ind, goal_state_ind, test_iter, test_output_prefix);
#endif
    

    grid::end_effector_positions_kernel<T><<<1,128>>>(d_eePos, d_xs, grid::NUM_JOINTS, (grid::robotModel<T> *) d_dynmem, 1);
    gpuErrchk(cudaMemcpy(h_eePos, d_eePos, 6*sizeof(T), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_eePos_goal, d_eePos_goal, 6*sizeof(T), cudaMemcpyDeviceToHost));
    cur_tracking_error_vec[0] = 0.0;
    for(uint32_t i=0; i < 3; i++){
        cur_tracking_error_vec[0] += abs(h_eePos[i] - h_eePos_goal[i]);
    }

    gato_plant::freeDynamicsConstMem<T>(d_dynmem);

    gpuErrchk(cudaFree(d_lambda));
    gpuErrchk(cudaFree(d_xu));
    gpuErrchk(cudaFree(d_eePos_goal));
    gpuErrchk(cudaFree(d_xu_old));

    gpuErrchk(cudaFree(d_eePos));

    #if TIME_LINSYS == 1 
        return std::make_tuple(linsys_times_vec[0], tracking_errors_vec[0], cur_tracking_error_vec[0]);
    #else
        return std::make_tuple(sqp_iters_vec[0], tracking_errors_vec[0], cur_tracking_error_vec[0]);
    #endif
}
