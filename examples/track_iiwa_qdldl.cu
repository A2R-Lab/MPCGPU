#pragma once
#include <fstream>
#include <vector>
#include <sstream>
#include <iostream>
#include <tuple>
#include <filesystem>
#include "qdldl.h"
#include "track.cuh"
#include "rbdfiles/rbd_plant.cuh"
#include "settings.cuh"
#include "utils/experiment.cuh"

int main(){

    constexpr uint32_t state_size = grid::NUM_JOINTS*2;
    constexpr uint32_t control_size = grid::NUM_JOINTS;
    constexpr uint32_t knot_points = KNOT_POINTS;
    const linsys_t timestep = .015625;

    const uint32_t traj_test_iters = TEST_ITERS;

    if(!std::is_same<QDLDL_float, linsys_t>::value){ std::cout << "GPU-PCG QDLDL type mismatch" << std::endl; exit(1); }

    print_test_config();
     // where to store test results — manually create this directory
    std::string output_directory_path = "/tmp/results/";

    const uint32_t recorded_states = 5;
    const uint32_t start_goal_combinations = recorded_states*recorded_states;

    char eePos_traj_file_name[100];
    char xu_traj_file_name[100];

    int start_state, goal_state;
    linsys_t *d_eePos_traj, *d_xu_traj, *d_xs;

    for(uint32_t ind = 0; ind < start_goal_combinations; ind++){

        start_state = ind % recorded_states;
        goal_state = ind / recorded_states;
        if(start_state == goal_state && start_state != 0){ continue; }
        std::cout << "start: " << start_state << " goal: " << goal_state << std::endl;

		float linsys_exit_tol = -1;
		std::vector<double> linsys_times;
		std::vector<uint32_t> sqp_iters;
		std::vector<toplevel_return_type> current_results;
		std::vector<float> tracking_errs;
		std::vector<float> cur_tracking_errs;
		double tot_final_tracking_err = 0;

		std::string test_output_prefix = output_directory_path  + std::to_string(KNOT_POINTS) + "_" + ( (LINSYS_SOLVE == 1) ? "PCG" : "QDLDL");
		printf("Logging test results to files with prefix %s \n", test_output_prefix.c_str()); 

		for (uint32_t single_traj_test_iter = 0; single_traj_test_iter < traj_test_iters; single_traj_test_iter++){

			// read in traj
			snprintf(eePos_traj_file_name, sizeof(eePos_traj_file_name), "examples/trajfiles/%d_%d_eepos.traj", start_state, goal_state);
			std::vector<std::vector<linsys_t>> eePos_traj2d = readCSVToVecVec<linsys_t>(eePos_traj_file_name);
			
			snprintf(xu_traj_file_name, sizeof(xu_traj_file_name), "examples/trajfiles/%d_%d_traj.csv", start_state, goal_state);
			std::vector<std::vector<linsys_t>> xu_traj2d = readCSVToVecVec<linsys_t>(xu_traj_file_name);
			
			if(eePos_traj2d.size() < knot_points){std::cout << "precomputed traj length < knotpoints, not implemented\n"; continue; }


			std::vector<linsys_t> h_eePos_traj;
			for (const auto& vec : eePos_traj2d) {
				h_eePos_traj.insert(h_eePos_traj.end(), vec.begin(), vec.end());
			}
			std::vector<linsys_t> h_xu_traj;
			for (const auto& xu_vec : xu_traj2d) {
				h_xu_traj.insert(h_xu_traj.end(), xu_vec.begin(), xu_vec.end());
			}

			gpuErrchk(cudaMalloc(&d_eePos_traj, h_eePos_traj.size()*sizeof(linsys_t)));
			gpuErrchk(cudaMemcpy(d_eePos_traj, h_eePos_traj.data(), h_eePos_traj.size()*sizeof(linsys_t), cudaMemcpyHostToDevice));
			
			gpuErrchk(cudaMalloc(&d_xu_traj, h_xu_traj.size()*sizeof(linsys_t)));
			gpuErrchk(cudaMemcpy(d_xu_traj, h_xu_traj.data(), h_xu_traj.size()*sizeof(linsys_t), cudaMemcpyHostToDevice));
			
			gpuErrchk(cudaMalloc(&d_xs, state_size*sizeof(linsys_t)));
			gpuErrchk(cudaMemcpy(d_xs, h_xu_traj.data(), state_size*sizeof(linsys_t), cudaMemcpyHostToDevice));

			std::tuple<std::vector<toplevel_return_type>, std::vector<linsys_t>, linsys_t> trackingstats = track<linsys_t, toplevel_return_type>(state_size, control_size, knot_points, 
				static_cast<uint32_t>(eePos_traj2d.size()), timestep, d_eePos_traj, d_xu_traj, d_xs, start_state, goal_state, single_traj_test_iter, linsys_exit_tol, test_output_prefix);
			
			current_results = std::get<0>(trackingstats);
			if (TIME_LINSYS == 1) {
				linsys_times.insert(linsys_times.end(), current_results.begin(), current_results.end());
			} else {
				sqp_iters.insert(sqp_iters.end(), current_results.begin(), current_results.end());
			}

			cur_tracking_errs = std::get<1>(trackingstats);
			tracking_errs.insert(tracking_errs.end(), cur_tracking_errs.begin(), cur_tracking_errs.end());

			tot_final_tracking_err += std::get<2>(trackingstats);
			


			gpuErrchk(cudaFree(d_xu_traj));
			gpuErrchk(cudaFree(d_eePos_traj));
			gpuErrchk(cudaFree(d_xs));
			gpuErrchk(cudaPeekAtLastError());
			
		}

		std::cout << "Completed at " << getCurrentTimestamp() << std::endl;
		std::cout << "\nRESULTS*************************************\n";
		std::cout << "exit tol: " << linsys_exit_tol << std::endl;
		std::cout << "\nTracking err";
		std::string trackingStats = printStats<float>(&tracking_errs, "trackingerr");
		std::cout << "Average final tracking err: " << tot_final_tracking_err / traj_test_iters << std::endl;
		std::string linsysOrSqpStats;
		if (TIME_LINSYS == 1)
		{
		std::cout << "\nLinsys times";
		linsysOrSqpStats = printStats<double>(&linsys_times, "linsystimes");
		}
		else
		{
		std::cout << "\nSqp iters";
		linsysOrSqpStats = printStats<uint32_t>(&sqp_iters, "sqpiters");
		}
		std::cout << "************************************************\n\n";


		// Specify the CSV file path
		const std::string csvFilePath = test_output_prefix + "_" + "overall_stats.csv";

		// Open the CSV file for writing
		std::ofstream csvFile(csvFilePath);
		if (!csvFile.is_open()) {
			std::cerr << "Error opening CSV file for writing." << std::endl;
			return 1;
		}

		// Write the header row
		csvFile << "Average,Std Dev, Min, Max, Median, Q1, Q3\n";

		// Write the data rows
		csvFile << getStatsString(trackingStats) << "\n";
		csvFile << getStatsString(linsysOrSqpStats) << "\n";

		// Close the CSV file
		csvFile.close();

        break;
    }




    return 0;
}