#include <fstream>
#include <vector>
#include <sstream>
#include <iostream>
#include <tuple>
#include <filesystem>
#include "dynamics/rbd_plant.cuh"
#include "settings.cuh"
#include "utils/experiment.cuh"
#include "gpu_pcg.cuh"
#include "../include/pcg/sqp_n.cuh"


int main(){
    constexpr uint32_t state_size = grid::NUM_JOINTS*2;
    constexpr uint32_t control_size = grid::NUM_JOINTS;
    constexpr uint32_t knot_points = KNOT_POINTS;
    const float timestep = .015625;
	const uint32_t solve_count = 3;

    // checks GPU space for pcg
    checkPcgOccupancy<linsys_t>((void *) pcg<linsys_t, state_size, knot_points>, PCG_NUM_THREADS, state_size, knot_points);    
	void *d_dynmem = gato_plant::initializeDynamicsConstMem<linsys_t>();
	const uint32_t traj_len = (state_size+control_size)*knot_points-control_size;

    linsys_t *d_eePos_traj, *d_xu_traj;
	linsys_t rho = 1e-3;

	uint32_t num_exit_vals = 5;
	linsys_t pcg_exit_vals[num_exit_vals];
	pcg_exit_vals[0] = 5e-6;
	pcg_exit_vals[1] = 7.5e-6;
	pcg_exit_vals[2] = 5e-6;
	pcg_exit_vals[3] = 2.5e-6;
	pcg_exit_vals[4] = 1e-6;
	

	linsys_t pcg_exit_tol = pcg_exit_vals[0];
	std::vector<toplevel_return_type> current_results;

	pcg_config<linsys_t> config;
	config.pcg_block = PCG_NUM_THREADS;
	config.pcg_exit_tol = pcg_exit_tol;
	config.pcg_max_iter = PCG_MAX_ITER;

	// read in traj
	auto eePos_traj2d = readCSVToVecVec<linsys_t>("examples/trajfiles/0_0_eepos.traj");
	auto xu_traj2d = readCSVToVecVec<linsys_t>("examples/trajfiles/0_0_traj.csv");

	std::vector<linsys_t> h_eePos_traj;
	for (const auto& vec : eePos_traj2d) {
		h_eePos_traj.insert(h_eePos_traj.end(), vec.begin(), vec.end());
	}
	std::vector<linsys_t> h_xu_traj;
	for (const auto& xu_vec : xu_traj2d) {
		h_xu_traj.insert(h_xu_traj.end(), xu_vec.begin(), xu_vec.end());
	}

	uint32_t lambda_size = state_size * knot_points * sizeof(linsys_t) * solve_count;
	uint32_t xu_size = traj_len * sizeof(linsys_t);
	uint32_t eePos_size = 6 * knot_points * sizeof(linsys_t);
	linsys_t *d_lambda;

	gpuErrchk(cudaMalloc(&d_eePos_traj, solve_count * eePos_size));
	gpuErrchk(cudaMalloc(&d_xu_traj, solve_count * xu_size));
	gpuErrchk(cudaMalloc(&d_lambda, lambda_size));
	gpuErrchk(cudaMemset(d_lambda, 0, lambda_size));

	gpuErrchk(cudaMemcpy(d_xu_traj, h_xu_traj.data(), xu_size, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_eePos_traj,  h_eePos_traj.data(), eePos_size, cudaMemcpyHostToDevice));

	for (uint32_t i = 1; i < solve_count; i++) {
		gpuErrchk(cudaMemcpy(d_xu_traj + i * traj_len, d_xu_traj, xu_size, cudaMemcpyDeviceToDevice));
		gpuErrchk(cudaMemcpy(d_eePos_traj + i * 6 * knot_points, d_eePos_traj, eePos_size, cudaMemcpyDeviceToDevice));
	}
	gpuErrchk(cudaDeviceSynchronize());

	sqpSolvePcg<linsys_t>(solve_count, state_size, control_size, knot_points, timestep, d_eePos_traj, d_lambda, d_xu_traj, d_dynmem, config, rho, 1e-3);
	
	linsys_t h_xu[xu_size * solve_count];
	linsys_t h_lambda[lambda_size / sizeof(linsys_t)];
	gpuErrchk(cudaMemcpy(h_xu, d_xu_traj, xu_size * solve_count, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_lambda, d_lambda, lambda_size, cudaMemcpyDeviceToHost));

	for (uint32_t i = 0; i < traj_len; i++) {
		for (uint32_t j = 0; j < solve_count; j++) {
			if (h_xu[i + j * traj_len] != h_xu_traj.at(i)) {
				printf("problem: %d, index: %d, %f %f \n", j, i, h_xu[i + j * traj_len], h_xu_traj.at(i));
			}
		}
	}

	for (uint32_t i = 0; i < solve_count; i++) {
		printf("problem: %d\n", i);
		for (uint32_t j = 0; j < state_size * knot_points; j++) {
			printf("%f,", h_lambda[j + i * state_size * knot_points]);
		}
		printf("\n");
	}
	gpuErrchk(cudaFree(d_xu_traj));
	gpuErrchk(cudaFree(d_eePos_traj));
	gpuErrchk(cudaFree(d_lambda));
	gpuErrchk(cudaPeekAtLastError());
    return 0;
}
