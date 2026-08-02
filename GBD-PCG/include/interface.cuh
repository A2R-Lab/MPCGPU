#pragma once
#include <iostream>
#include <stdint.h>
#include "gpuassert.cuh"
#include "types.cuh"
#include "pcg.cuh"

template <typename T>
uint32_t solvePCG(
    csr_t<T> *h_S, 
    csr_t<T> *h_Pinv, 
    T *h_gamma, 
    T *h_lambda, 
    unsigned stateSize, 
    unsigned knotPoints, 
    struct pcg_config<T> *config)
{
    std::cout << "NOT IMPLEMENTED" << std::endl;
    exit(12);
}


/* TODO: have a interface for accepting h_S in other formats*/
template <typename T>
uint32_t solvePCG(
    T *h_S, 
    T *h_gamma, 
    T *h_lambda, 
    unsigned stateSize, 
    unsigned knotPoints, 
    struct pcg_config<T> *config)
{
	if (!config->empty_pinv)
		printf("This api can only be called with no preconditioner\n");

	const uint32_t states_sq = stateSize*stateSize;
    /* Create device memory d_s, d_Pinv, d_gamma, d_lambda, d_r, d_p, d_v_temp
	d_eta_new_temp */
	T *d_S, *d_gamma, *d_lambda;
	gpuErrchk(cudaMalloc(&d_lambda, stateSize*knotPoints*sizeof(T)));
	gpuErrchk(cudaMalloc(&d_S, 3*states_sq*knotPoints*sizeof(T)));
    gpuErrchk(cudaMalloc(&d_gamma, stateSize*knotPoints*sizeof(T)));


	T *d_Pinv;
    gpuErrchk(cudaMalloc(&d_Pinv, 3*states_sq*knotPoints*sizeof(T)));
    
    /*   PCG vars   */
    T  *d_r, *d_p, *d_v_temp, *d_eta_new_temp;
    gpuErrchk(cudaMalloc(&d_r, stateSize*knotPoints*sizeof(T)));
    gpuErrchk(cudaMalloc(&d_p, stateSize*knotPoints*sizeof(T)));
    gpuErrchk(cudaMalloc(&d_v_temp, knotPoints*sizeof(T)));
    gpuErrchk(cudaMalloc(&d_eta_new_temp, knotPoints*sizeof(T)));


	/* Copy s, gamma, lambda*/
	gpuErrchk(cudaMemcpy(d_S, h_S, 3 * states_sq * knotPoints * sizeof(T), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_lambda, h_lambda, stateSize * knotPoints * sizeof(T), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_gamma, h_gamma, stateSize * knotPoints * sizeof(T), cudaMemcpyHostToDevice));


	solvePCG(stateSize, knotPoints,
                  d_S,
                  d_Pinv,
                  d_gamma,
                  d_lambda,
                  d_r,
                  d_p,
                  d_v_temp,
                  d_eta_new_temp,
                  config);


    /* Copy data back */
	gpuErrchk(cudaMemcpy(h_S, d_S, 3 * states_sq * knotPoints * sizeof(T), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_lambda, d_lambda, stateSize * knotPoints * sizeof(T), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_gamma, d_gamma, stateSize * knotPoints * sizeof(T), cudaMemcpyDeviceToHost));

	cudaFree(d_lambda);
	cudaFree(d_S);
	cudaFree(d_gamma);
	cudaFree(d_Pinv);
	cudaFree(d_r);
	cudaFree(d_p);
	cudaFree(d_v_temp);
	cudaFree(d_eta_new_temp);

	return 1;
}


template <typename T>
uint32_t solvePCG(const uint32_t state_size,
                  const uint32_t knot_points,
                  T *d_S,
                  T *d_Pinv,
                  T *d_gamma,
                  T *d_lambda,
                  T *d_r,
                  T *d_p,
                  T *d_v_temp,
                  T *d_eta_new_temp,
                  struct pcg_config<T> *config)
{
    uint32_t *d_pcg_iters;
    gpuErrchk(cudaMalloc(&d_pcg_iters, sizeof(uint32_t)));
    bool *d_pcg_exit;
    gpuErrchk(cudaMalloc(&d_pcg_exit, sizeof(bool)));
    
    void *pcg_kernel = (void *) pcg<T, STATE_SIZE, KNOT_POINTS>;

    // checkPcgOccupancy<T>(pcg_kernel, config->pcg_block, state_size, knot_points);
    void *kernelArgs[] = {
        (void *)&d_S,
        (void *)&d_Pinv,
        (void *)&d_gamma, 
        (void *)&d_lambda,
        (void *)&d_r,
        (void *)&d_p,
        (void *)&d_v_temp,
        (void *)&d_eta_new_temp,
        (void *)&d_pcg_iters,
        (void *)&d_pcg_exit,
        (void *)&config->pcg_max_iter,
        (void *)&config->pcg_exit_tol,
        (void *)&config->pcg_rel_tol,
		(void *)&config->empty_pinv
    };
    uint32_t h_pcg_iters;

    size_t ppcg_kernel_smem_size = pcgSharedMemSize<T>(state_size, knot_points);

    gpuErrchk(cudaLaunchCooperativeKernel(pcg_kernel, knot_points, pcg_constants::DEFAULT_BLOCK, kernelArgs, ppcg_kernel_smem_size));    
    gpuErrchk(cudaPeekAtLastError());


    gpuErrchk(cudaMemcpy(&h_pcg_iters, d_pcg_iters, sizeof(uint32_t), cudaMemcpyDeviceToHost));



    gpuErrchk(cudaFree(d_pcg_iters));
	gpuErrchk(cudaFree(d_pcg_exit));

    return h_pcg_iters;
}