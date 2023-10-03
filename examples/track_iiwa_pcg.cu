#include <fstream>
#include <vector>
#include <sstream>
#include <iostream>
#include <tuple>
#include <filesystem>
#include "track.cuh"
#include "rbdfiles/rbd_plant.cuh"
#include "settings.cuh"
#include "utils/experiment.cuh"
#include "gpu_pcg.cuh"


int main(){

	// problem parameters
    constexpr uint32_t state_size = grid::NUM_JOINTS*2;
    constexpr uint32_t control_size = grid::NUM_JOINTS;
    constexpr uint32_t knot_points = KNOT_POINTS;
    const linsys_t timestep = .015625;

	// test parameters
    const uint32_t traj_test_iters = TEST_ITERS;
    char eePos_traj_file_name[] = "examples/precomputedTrajectories/0_0_eepos.csv";
    char xu_traj_file_name[] = "examples/precomputedTrajectories/0_0_eepos.csv";
    std::string output_directory_path = "/tmp/results/";
    print_test_config();



    // checks GPU space for pcg
    checkPcgOccupancy<linsys_t>((void *) pcg<linsys_t, state_size, knot_points>, PCG_NUM_THREADS, state_size, knot_points);    




	// setting pcg exit tolerances that will be tested
	uint32_t num_exit_vals = 5;
	float pcg_exit_vals[num_exit_vals];
	
	
	if(knot_points==32){
		pcg_exit_vals[0] = 1e-7;
		pcg_exit_vals[1] = 7.5e-6;
		pcg_exit_vals[2] = 5e-6;
		pcg_exit_vals[3] = 2.5e-6;
		pcg_exit_vals[4] = 1e-6;
	}
	else if(knot_points==64){
		pcg_exit_vals[0] = 5e-5;
		pcg_exit_vals[1] = 7.5e-5;
		pcg_exit_vals[2] = 5e-5;
		pcg_exit_vals[3] = 2.5e-5;
		pcg_exit_vals[4] = 1e-5;
	}
	else{
		pcg_exit_vals[0] = 1e-5;
		pcg_exit_vals[1] = 5e-5;
		pcg_exit_vals[2] = 1e-4;
		pcg_exit_vals[3] = 5e-4;
		pcg_exit_vals[4] = 1e-3;
	}

	

	std::tuple<std::vector<toplevel_return_type>, std::vector<linsys_t>, linsys_t> trackingstats;
    linsys_t *d_eePos_traj, *d_xu_traj, *d_xs;


	// loop over PCG exit tolerance values
	for (uint32_t pcg_exit_ind = 0; pcg_exit_ind < num_exit_vals; pcg_exit_ind++){

		// setup
		float pcg_exit_tol = pcg_exit_vals[pcg_exit_ind];
		std::vector<double> linsys_times;
		std::vector<uint32_t> sqp_iters;
		std::vector<toplevel_return_type> current_results;
		std::vector<float> tracking_errs;
		std::vector<float> cur_tracking_errs;
		double tot_final_tracking_err;
		tot_final_tracking_err = 0;

		std::string test_output_prefix = output_directory_path + std::to_string(KNOT_POINTS) + "_" + ( (LINSYS_SOLVE == 1) ? "PCG" : "QDLDL") + "_" + std::to_string(pcg_exit_tol);
		printf("Logging test results to files with prefix %s \n", test_output_prefix.c_str()); 


		// each trajectory is tested traj_test_iters times
		for (uint32_t single_traj_test_iter = 0; single_traj_test_iter < traj_test_iters; single_traj_test_iter++){

			//
			// boring setup
			//
		
			
			std::vector<std::vector<linsys_t>> eePos_traj2d = readCSVToVecVec<linsys_t>(eePos_traj_file_name);
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




			// run tracking experiment
			trackingstats = track<linsys_t, toplevel_return_type>(
				state_size,
				control_size,
				knot_points,
				static_cast<uint32_t>(eePos_traj2d.size()), 			// length of precomputed trajectory in knot points
				timestep, 												// timestep in seconds
				d_eePos_traj, 											// device memory region for end effector states
				d_xu_traj, 												// device memory region for joint states
				d_xs, 													// device memory region for start state
				pcg_exit_tol,	
				single_traj_test_iter,									// test iteration (used in naming data files)										 
				test_output_prefix
			);


			// collect and store data
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
		std::cout << "Exit tol: " << pcg_exit_tol << std::endl;
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
		
    }




    return 0;
}