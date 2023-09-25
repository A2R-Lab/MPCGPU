#pragma once
#include <vector>
#include <numeric>
#include <algorithm>
#include <cstdint>
#include <cublas_v2.h>
#include <math.h>
#include <cmath>
#include <random>
#include <iomanip>
#include <cuda_runtime.h>
#include <tuple>
#include <time.h>
#include "types.cuh"

#if PCG_SOLVE
#include "linsys_solvers/sqp_pcg.cuh"
#else 
#include "linsys_solvers/sqp_qdldl.cuh"
#endif   


template <typename T>
auto sqpSolve(const uint32_t state_size, const uint32_t control_size, const uint32_t knot_points, float timestep, T *d_eePos_traj, T *d_lambda, T *d_xu, void *d_dynMem_const, pcg_config& config, T &rho, T rho_reset){
#if PCG_SOLVE
	return sqpSolvePcg<T>(state_size, control_size, knot_points, timestep, d_eePos_traj, d_lambda, d_xu, d_dynMem_const, config, rho, rho_reset);
#else 
	return sqpSolveQdldl<T>(state_size, control_size, knot_points, timestep, d_eePos_traj, d_lambda, d_xu, d_dynMem_const, rho, rho_reset);
#endif        
}
