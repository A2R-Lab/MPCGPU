#pragma once
#include <stdint.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "types.cuh"
#include "gpuassert.cuh"
#include "utils.cuh"
#include "glass.cuh"


namespace cgrps = cooperative_groups;

template <typename T>
size_t pcgSharedMemSize(uint32_t state_size, uint32_t knot_points){
    return sizeof(T) * max(
                        (2*3*state_size*state_size + 
                        10 * state_size + 
                        2*max(state_size, knot_points)),
                        (9 * state_size*state_size));
}


template <typename T>
bool checkPcgOccupancy(void* kernel, dim3 block, uint32_t state_size, uint32_t knot_points){
    
    const uint32_t smem_size = pcgSharedMemSize<T>(state_size, knot_points);
    int dev = 0;
    
    cudaDeviceProp deviceProp; 
    cudaGetDeviceProperties(&deviceProp, dev);
    
    int supportsCoopLaunch = 0; 
    gpuErrchk(cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev));
    if(!supportsCoopLaunch){
        printf("[Error] Device does not support Cooperative Threads\n");
        exit(5);
    }
    
    int numProcs = deviceProp.multiProcessorCount; 
    int numBlocksPerSm;
    gpuErrchk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, kernel, block.x*block.y*block.z, smem_size));

    if((int) knot_points > numProcs*numBlocksPerSm){
        printf("Too many knot points ([%d]). Device supports [%d] active blocks, over [%d] SMs.\n", knot_points, numProcs*numBlocksPerSm, numProcs);
        exit(6);
    }

    return true;
}




template <typename T, uint32_t state_size, uint32_t knot_points>
__global__
void pcg(
         T *d_S, 
         T *d_Pinv, 
         T *d_gamma,  				
         T *d_lambda, 
         T  *d_r, 
         T  *d_p, 
         T *d_v_temp, 
         T *d_eta_new_temp,
         uint32_t *d_iters, 
         bool *d_max_iter_exit,
         uint32_t max_iter, 
         T exit_tol)
{   

    const cgrps::thread_block block = cgrps::this_thread_block();	 
    const cgrps::grid_group grid = cgrps::this_grid();
    const uint32_t block_id = blockIdx.x;
    const uint32_t block_dim = blockDim.x;
    const uint32_t thread_id = threadIdx.x;
    const uint32_t block_x_statesize = block_id * state_size;
    const uint32_t states_sq = state_size * state_size;

    extern __shared__ T s_temp[];
    


    T  *s_S = s_temp;
    T  *s_Pinv = s_S +3*states_sq;
    T  *s_gamma = s_Pinv + 3*states_sq;
    T  *s_scratch = s_gamma + state_size;
    T *s_lambda = s_scratch;
    T *s_r_tilde = s_lambda + 3*state_size;
    T  *s_upsilon = s_r_tilde + state_size;
    T  *s_v_b = s_upsilon + state_size;
    T  *s_eta_new_b = s_v_b + max(knot_points, state_size);
    T  *s_r = s_eta_new_b + max(knot_points, state_size);
    T  *s_p = s_r + 3*state_size;
    T  *s_r_b = s_r + state_size;
    T  *s_p_b = s_p + state_size;
    T *s_lambda_b = s_lambda + state_size;

    uint32_t iter;
    T alpha, beta, eta, eta_new;

    bool max_iter_exit = true;

    // populate shared memory. Zero the absent L (block 0) / R (last block) strips so
    // bdmv runs one uniform full-width [L|D|R] matvec (glass::gemv) with no boundary
    // cases — the zeroed strip multiplies the zeroed halo pad slot (see loadbdVec).
    for (unsigned ind = thread_id; ind < 3*states_sq; ind += block_dim){
        if(block_id == 0 && ind < states_sq){ s_S[ind] = static_cast<T>(0); s_Pinv[ind] = static_cast<T>(0); continue; }
        if(block_id == knot_points-1 && ind >= 2*states_sq){ s_S[ind] = static_cast<T>(0); s_Pinv[ind] = static_cast<T>(0); continue; }

        s_S[ind] = d_S[block_id*states_sq*3 + ind];
        s_Pinv[ind] = d_Pinv[block_id*states_sq*3 + ind];
    }
    glass::copy<T>(state_size, &d_gamma[block_x_statesize], s_gamma);


    //
    // PCG
    //

    // r = gamma - S * lambda
    loadbdVec<T, state_size, knot_points-1>(s_lambda, block_id, &d_lambda[block_x_statesize]);
    __syncthreads();
    bdmv<T>(s_r_b, s_S, s_lambda, state_size, knot_points-1,  block_id);
    __syncthreads();
    for (unsigned ind = thread_id; ind < state_size; ind += block_dim){
        s_r_b[ind] = s_gamma[ind] - s_r_b[ind];
        d_r[block_x_statesize + ind] = s_r_b[ind]; 
    }
    
    grid.sync(); //-------------------------------------

    // r_tilde = Pinv * r
    loadbdVec<T, state_size, knot_points-1>(s_r, block_id, &d_r[block_x_statesize]);
    __syncthreads();
    bdmv<T>(s_r_tilde, s_Pinv, s_r, state_size, knot_points-1,  block_id);
    __syncthreads();
    
    // p = r_tilde
    for (unsigned ind = thread_id; ind < state_size; ind += block_dim){
        s_p_b[ind] = s_r_tilde[ind];
        d_p[block_x_statesize + ind] = s_p_b[ind]; 
    }


    // eta = r * r_tilde  (dot_lowmem leaves s_r_b / s_r_tilde intact; result in s_eta_new_b[0])
    glass::dot_lowmem<T>(state_size, s_r_b, s_r_tilde, s_eta_new_b);
    if(thread_id == 0){ d_eta_new_temp[block_id] = s_eta_new_b[0]; }
    grid.sync(); //-------------------------------------
    // cross-block sum: load the per-block partials into shared, then in-place reduce → [0]
    glass::copy<T>(knot_points, d_eta_new_temp, s_eta_new_b);
    __syncthreads();
    glass::reduce<T>(knot_points, s_eta_new_b);
    __syncthreads();
    eta = s_eta_new_b[0];

    // Converged-start / zero-RHS guard (mirrors glass::pcg solve.cuh). If the initial preconditioned
    // residual is already below tolerance, lambda (the warm start) IS the solution — skip the loop.
    // Without this, iter 0 computes alpha = eta / (pᵀSp) = 0/0 → NaN whenever the residual starts at
    // ~0: regulation (γ≈0), or any well-tracked warm start (this is the offset-6 NaN on moving refs).
    // All blocks share the grid-reduced eta, so the early return is uniform — no grid.sync mismatch.
    if(abs(eta) < exit_tol){
        if(block_id == 0 && thread_id == 0){ d_iters[0] = 0; d_max_iter_exit[0] = false; }
        __syncthreads();
        glass::copy<T>(state_size, s_lambda_b, &d_lambda[block_x_statesize]);
        grid.sync();
        return;
    }


    // MAIN PCG LOOP

    for(iter = 0; iter < max_iter; iter++){

        // upsilon = S * p
        loadbdVec<T, state_size, knot_points-1>(s_p, block_id, &d_p[block_x_statesize]);
        __syncthreads();
        bdmv<T>(s_upsilon,  s_S, s_p,state_size, knot_points-1, block_id);
        __syncthreads();

        // alpha = eta / (p * upsilon)
        glass::dot_lowmem<T>(state_size, s_p_b, s_upsilon, s_v_b);
        __syncthreads();
        if(thread_id == 0){ d_v_temp[block_id] = s_v_b[0]; }
        grid.sync(); //-------------------------------------
        glass::copy<T>(knot_points, d_v_temp, s_v_b);
        __syncthreads();
        glass::reduce<T>(knot_points, s_v_b);
        __syncthreads();
        alpha = eta / s_v_b[0];
        // lambda = lambda + alpha * p
        // r = r - alpha * upsilon
        for(uint32_t ind = thread_id; ind < state_size; ind += block_dim){
            s_lambda_b[ind] += alpha * s_p_b[ind];
            s_r_b[ind] -= alpha * s_upsilon[ind];
            d_r[block_x_statesize + ind] = s_r_b[ind];
        }

        grid.sync(); //-------------------------------------

        // r_tilde = Pinv * r
        loadbdVec<T, state_size, knot_points-1>(s_r, block_id, &d_r[block_x_statesize]);
        __syncthreads();
        bdmv<T>(s_r_tilde, s_Pinv, s_r, state_size, knot_points-1, block_id);
        __syncthreads();

        // eta_new = r * r_tilde  (inputs preserved; result in s_eta_new_b[0])
        glass::dot_lowmem<T>(state_size, s_r_b, s_r_tilde, s_eta_new_b);
        __syncthreads();
        if(thread_id == 0){ d_eta_new_temp[block_id] = s_eta_new_b[0]; }
        grid.sync(); //-------------------------------------
        glass::copy<T>(knot_points, d_eta_new_temp, s_eta_new_b);
        __syncthreads();
        glass::reduce<T>(knot_points, s_eta_new_b);
        __syncthreads();
        eta_new = s_eta_new_b[0];

        if(abs(eta_new) < exit_tol){ iter++; max_iter_exit = false; break; }

        // beta = eta_new / eta
        // eta = eta_new
        beta = eta_new / eta;
        eta = eta_new;

        // p = r_tilde + beta*p
        for(uint32_t ind = thread_id; ind < state_size; ind += block_dim){
            s_p_b[ind] = s_r_tilde[ind] + beta*s_p_b[ind];
            d_p[block_x_statesize + ind] = s_p_b[ind];
        }
        grid.sync(); //-------------------------------------
    }


    // save output
    if(block_id == 0 && thread_id == 0){ d_iters[0] = iter; d_max_iter_exit[0] = max_iter_exit; }
    
    __syncthreads();
    glass::copy<T>(state_size, s_lambda_b, &d_lambda[block_x_statesize]);

    grid.sync();
}