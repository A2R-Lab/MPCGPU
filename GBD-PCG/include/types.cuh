#pragma once

#include <cstdint>              // for uint32_t
#include <cuda_runtime.h>       // for dim3
#include "constants.cuh"

template <typename T>
struct csr_t{
    uint32_t *row_ptr;
    uint32_t *col_ind;
    T *val;
    uint32_t rows;
    uint32_t cols;
    uint32_t nnz;
};


template <typename T>
struct pcg_config{
	
    T pcg_exit_tol;   // absolute floor on the preconditioned residual eta = r^T Pinv r
    T pcg_rel_tol;    // relative tol: stop when |eta| < pcg_exit_tol + pcg_rel_tol*|eta_init|
    uint32_t pcg_max_iter;

    dim3 pcg_grid;
    dim3 pcg_block;

	int empty_pinv;

    pcg_config(T    exit_tol = pcg_constants::DEFAULT_EPSILON<T>,
               T    rel_tol = static_cast<T>(1e-5),
               uint32_t max_iter = pcg_constants::DEFAULT_MAX_PCG_ITER,
               dim3     grid = pcg_constants::DEFAULT_GRID, 
               dim3     block = pcg_constants::DEFAULT_BLOCK,
			   int 		empty_pinv = 1)             
        : pcg_exit_tol(exit_tol), pcg_rel_tol(rel_tol), pcg_max_iter(max_iter), pcg_grid(grid), pcg_block(block), empty_pinv(empty_pinv) {}
};