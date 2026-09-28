#pragma once
#include <vector>
#include "common/workspace.cuh"
#include <type_traits>
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
#include "qdldl.h"
#include "qdldl/linsys_setup.cuh"
#include "merit.cuh"
#include "settings.cuh"
#include "kkt.cuh"
#include "dz.cuh"


__host__
void qdldl_solve_schur(const QDLDL_int An,
					   QDLDL_int *h_col_ptr, QDLDL_int *h_row_ind, QDLDL_float *Ax, QDLDL_float *b, 
					   QDLDL_float *h_lambda,
					   QDLDL_int *Lp, QDLDL_int *Li, QDLDL_float *Lx, QDLDL_float *D, QDLDL_float *Dinv, QDLDL_int *Lnz, QDLDL_int *etree, QDLDL_bool *bwork, QDLDL_int *iwork, QDLDL_float *fwork){

	



    QDLDL_int i;

	const QDLDL_int *Ap = h_col_ptr;
	const QDLDL_int *Ai = h_row_ind;

    //data for L and D factors
	QDLDL_int Ln = An;


	//Data for results of A\b
	QDLDL_float *x = h_lambda;

	if (QDLDL_factor(An,Ap,Ai,Ax,Lp,Li,Lx,D,Dinv,Lnz,etree,bwork,iwork,fwork) < 0)
        throw std::runtime_error("QDLDL numerical factorization failed");

	for(i=0;i < Ln; i++) x[i] = b[i];

	QDLDL_solve(Ln,Lp,Li,Lx,Dinv,x);
}


template <typename T>
auto sqpSolveQdldl(uint32_t state_size, uint32_t control_size, uint32_t knot_points, float timestep, T *d_eePos_traj, T *d_lambda, T *d_xu, void *d_dynMem_const, T &rho, T rho_reset, T *d_xs_goal = nullptr, mpcgpu::SqpWorkspace* reuse = nullptr){
    if (state_size != 14 || control_size != 7 || knot_points != KNOT_POINTS || knot_points < 2)
        throw std::invalid_argument("MPCGPU requires iiwa14 dimensions and the compiled horizon");
    struct timespec sqp_solve_start, sqp_solve_end;
    gpuErrchk(cudaDeviceSynchronize());
#ifndef MPCGPU_CORRECTNESS
    clock_gettime(CLOCK_MONOTONIC, &sqp_solve_start);
#endif
    std::unique_ptr<mpcgpu::SqpWorkspace> owned;
    if (!reuse) { owned = std::make_unique<mpcgpu::SqpWorkspace>(state_size, control_size, knot_points); reuse = owned.get(); }
    auto& workspace = *reuse;
    workspace.begin(state_size, control_size, knot_points, 1);

    
    static_assert(std::is_same_v<T, QDLDL_float>, "QDLDL and solver precision must match");
    // data storage
    std::vector<int> linsys_iter_vec;
    std::vector<bool> linsys_exit_vec;
    std::vector<double> linsys_time_vec;
    bool sqp_time_exit = 1;     // for data recording, not a flag
    


    // sqp timing
    // Clock starts at entry; fresh workspace construction is part of this call.


    const uint32_t states_sq = state_size*state_size;
    const uint32_t states_p_controls = state_size * control_size;
    const uint32_t controls_sq = control_size * control_size;
    const uint32_t states_s_controls = state_size + control_size;
    const uint32_t KKT_G_DENSE_SIZE_BYTES = static_cast<uint32_t>(((states_sq+controls_sq)*knot_points-controls_sq)*sizeof(T));
    const uint32_t KKT_C_DENSE_SIZE_BYTES = static_cast<uint32_t>((states_sq+states_p_controls)*(knot_points-1)*sizeof(T));
    const uint32_t KKT_g_SIZE_BYTES       = static_cast<uint32_t>(((state_size+control_size)*knot_points-control_size)*sizeof(T));
    const uint32_t KKT_c_SIZE_BYTES       =   static_cast<uint32_t>((state_size*knot_points)*sizeof(T));     
    const uint32_t DZ_SIZE_BYTES          =   static_cast<uint32_t>((states_s_controls*knot_points-control_size)*sizeof(T));


    // line search things
    const float mu = 10.0f;
    const uint32_t num_alphas = 8;
    T h_merit_news[num_alphas];
    void *ls_merit_kernel = (void *) ls_gato_compute_merit<T>;
    const size_t merit_smem_size = get_merit_smem_size<T>(state_size, control_size);
    T h_merit_initial, min_merit;
    T alphafinal;
    T delta_merit_iter = 0;
    T delta_merit_total = 0;
    uint32_t line_search_step = 0;


    auto& streams = workspace.streams;
    auto handle = workspace.handle;
    uint32_t sqp_iter = 0;



    T *d_merit_initial, *d_merit_news, *d_merit_temp,
          *d_G_dense, *d_C_dense, *d_g, *d_c, *d_Ginv_dense,
          *d_S, *d_gamma,
          *d_dz,
          *d_xs;

    
    T drho = 1.0;
    T rho_factor = RHO_FACTOR;
    T rho_max = RHO_MAX;
    T rho_min = RHO_MIN;

    


    d_G_dense = workspace.device<T>((KKT_G_DENSE_SIZE_BYTES) / sizeof(T));
    d_C_dense = workspace.device<T>((KKT_C_DENSE_SIZE_BYTES) / sizeof(T));
    d_g = workspace.device<T>((KKT_g_SIZE_BYTES) / sizeof(T));
    d_c = workspace.device<T>((KKT_c_SIZE_BYTES) / sizeof(T));
    d_Ginv_dense = d_G_dense;

    d_S = workspace.device<T>((3*states_sq*knot_points*sizeof(T)) / sizeof(T));
    d_gamma = workspace.device<T>((state_size*knot_points*sizeof(T)) / sizeof(T));
    gpuErrchk(cudaPeekAtLastError());

    
    d_dz = workspace.device<T>((DZ_SIZE_BYTES) / sizeof(T));
    d_xs = workspace.device<T>((state_size*sizeof(T)) / sizeof(T));
    gpuErrchk(cudaMemcpy(d_xs, d_xu,  state_size*sizeof(T), cudaMemcpyDeviceToDevice));
    d_merit_news = workspace.device<T>((8*sizeof(T)) / sizeof(T));
    d_merit_temp = workspace.device<T>((8*knot_points*sizeof(T)) / sizeof(T));
    // linsys iterates

    d_merit_initial = workspace.device<T>((sizeof(T)) / sizeof(T));
    gpuErrchk(cudaMemset(d_merit_initial, 0, sizeof(T)));
    



    const int nnz = (knot_points-1)*states_sq + knot_points*(((state_size+1)*state_size)/2);
    
    QDLDL_float* h_lambda = workspace.host<QDLDL_float>(state_size*knot_points);
    QDLDL_float* h_gamma = workspace.host<QDLDL_float>(state_size*knot_points);
    QDLDL_int* h_col_ptr = workspace.host<QDLDL_int>(state_size*knot_points+1);
    QDLDL_int* h_row_ind = workspace.host<QDLDL_int>(nnz);
    QDLDL_float* h_val = workspace.host<QDLDL_float>(nnz);
    
    QDLDL_int *d_row_ind, *d_col_ptr;
    QDLDL_float *d_val, *d_lambda_double;
    d_col_ptr = workspace.device<QDLDL_int>(((state_size*knot_points+1)*sizeof(QDLDL_int)) / sizeof(QDLDL_int));
    d_row_ind = workspace.device<QDLDL_int>((nnz*sizeof(QDLDL_int)) / sizeof(QDLDL_int));
	d_val = workspace.device<QDLDL_float>((nnz*sizeof(QDLDL_float)) / sizeof(QDLDL_float));
	d_lambda_double = workspace.device<QDLDL_float>(((state_size*knot_points)*sizeof(QDLDL_float)) / sizeof(QDLDL_float));
    
    // fill col ptr and row ind, these won't change 
    if (!workspace.symbolic_ready) {
        prep_csr<<<knot_points, 64>>>(state_size, knot_points, d_col_ptr, d_row_ind);
        gpuErrchk(cudaMemcpy(h_col_ptr, d_col_ptr, (state_size*knot_points+1)*sizeof(QDLDL_int), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(h_row_ind, d_row_ind, nnz*sizeof(QDLDL_int), cudaMemcpyDeviceToHost));
    }

    
    const QDLDL_int An = state_size*knot_points;

    // Q things
    QDLDL_int  sumLnz;
    QDLDL_int *etree;
	QDLDL_int *Lnz;
    etree = workspace.host<QDLDL_int>(An);
	Lnz = workspace.host<QDLDL_int>(An);
    
    QDLDL_int *Lp;
	QDLDL_float *D;
	QDLDL_float *Dinv;
    Lp = workspace.host<QDLDL_int>((An+1));
	D = workspace.host<QDLDL_float>(An);
	Dinv = workspace.host<QDLDL_float>(An);

    //working data for factorisation
	QDLDL_int   *iwork;
	QDLDL_bool  *bwork;
	QDLDL_float *fwork;
    iwork = workspace.host<QDLDL_int>((3*An));
	bwork = workspace.host<QDLDL_bool>(An);
	fwork = workspace.host<QDLDL_float>(An);

    if (!workspace.symbolic_ready) {
        sumLnz = QDLDL_etree(An,h_col_ptr,h_row_ind,iwork,Lnz,etree);
        if (sumLnz < 0) throw std::runtime_error("QDLDL symbolic factorization failed");
        workspace.symbolic_nnz = sumLnz;
        workspace.symbolic_ready = true;
    }
    sumLnz = workspace.symbolic_nnz;
    
    QDLDL_int *Li;
	QDLDL_float *Lx;
    Li = workspace.host<QDLDL_int>(sumLnz);
	Lx = workspace.host<QDLDL_float>(sumLnz);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#if TIME_LINSYS == 1
    struct timespec linsys_start, linsys_end;
    double linsys_time;
#endif
#if CONST_UPDATE_FREQ && !defined(MPCGPU_CORRECTNESS)
    struct timespec sqp_cur;
    auto sqpTimecheck = [&]() {
        clock_gettime(CLOCK_MONOTONIC, &sqp_cur);
        return time_delta_us_timespec(sqp_solve_start,sqp_cur) > SQP_MAX_TIME_US;
    };
#else
    auto sqpTimecheck = [&]() { return false; };
#endif


    // Deterministic merit: per-knot temp (no atomicAdd) + fixed-order reduce.
    compute_merit<T><<<knot_points, MERIT_THREADS, merit_smem_size>>>(
        state_size, control_size, knot_points,
        d_xu,
        d_eePos_traj,
        static_cast<T>(10),
        timestep,
        d_dynMem_const,
        d_merit_temp,
        d_xs_goal
    );
    reduce_merit<T><<<1, MERIT_THREADS>>>(knot_points, d_merit_temp, d_merit_initial);
    gpuErrchk(cudaMemcpyAsync(&h_merit_initial, d_merit_initial, sizeof(T), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaPeekAtLastError());

    // gpuErrchk(cudaDeviceSynchronize());
    // std::cout << "initial merit " << h_merit_initial << std::endl;
    // exit(0);

    //
    //      SQP LOOP
    //
    for(uint32_t sqpiter = 0; sqpiter < SQP_MAX_ITER; sqpiter++){
        
        generate_kkt_submatrices<T, MPCGPU_INTEGRATOR><<<knot_points, KKT_THREADS, 2 * get_kkt_smem_size<T>(state_size, control_size)>>>(
            state_size,
            control_size,
            knot_points,
            d_G_dense, 
            d_C_dense, 
            d_g, 
            d_c,
            d_dynMem_const,
            timestep,
            d_eePos_traj,
            d_xs,
            d_xu,
            d_xs_goal
        );
        gpuErrchk(cudaPeekAtLastError());
        if (sqpTimecheck()){ break; }


        form_schur_system_qdldl<T>(state_size, control_size, knot_points, d_G_dense, d_C_dense, d_g, d_c, d_val, d_gamma, rho);
        gpuErrchk(cudaPeekAtLastError());
        if (sqpTimecheck()){ break; }

    #if TIME_LINSYS == 1
        gpuErrchk(cudaDeviceSynchronize());
        if (sqpTimecheck()){ break; }
        clock_gettime(CLOCK_MONOTONIC, &linsys_start);
    #endif // #if TIME_LINSYS


        gpuErrchk(cudaMemcpy(h_val, d_val, (nnz)*sizeof(T), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(h_gamma, d_gamma, (state_size*knot_points)*sizeof(T), cudaMemcpyDeviceToHost))

        qdldl_solve_schur(An, h_col_ptr, h_row_ind, h_val, h_gamma, h_lambda, Lp, Li, Lx, D, Dinv, Lnz, etree, bwork, iwork, fwork);
        
        gpuErrchk(cudaMemcpy(d_lambda, h_lambda, (state_size*knot_points)*sizeof(T), cudaMemcpyHostToDevice));


    #if TIME_LINSYS == 1
        gpuErrchk(cudaDeviceSynchronize());
        clock_gettime(CLOCK_MONOTONIC, &linsys_end);
        
        linsys_time = time_delta_us_timespec(linsys_start, linsys_end);
        linsys_time_vec.push_back(linsys_time);
    #endif // #if TIME_LINSYS
        
        if (sqpTimecheck()){ break; }
        
        // recover dz
        compute_dz(
            state_size,
            control_size,
            knot_points,
            d_Ginv_dense, 
            d_C_dense, 
            d_g, 
            d_lambda, 
            d_dz
        );
        gpuErrchk(cudaPeekAtLastError());
        if (sqpTimecheck()){ break; }
        

        // line search
        for(uint32_t p = 0; p < num_alphas; p++){
            void *kernelArgs[] = {
                (void *)&state_size,
                (void *)&control_size,
                (void *)&knot_points,
                (void *)&d_xs,
                (void *)&d_xu,
                (void *)&d_eePos_traj,
                (void *)&mu, 
                (void *)&timestep,
                (void *)&d_dynMem_const,
                (void *)&d_dz,
                (void *)&p,
                (void *)&d_merit_news,
                (void *)&d_merit_temp,
                (void *)&d_xs_goal
            };
            gpuErrchk(cudaLaunchCooperativeKernel(ls_merit_kernel, knot_points, MERIT_THREADS, kernelArgs, get_merit_smem_size<T>(state_size, knot_points), streams[p]));
        }
        if (sqpTimecheck()){ break; }
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        
        
        cudaMemcpy(h_merit_news, d_merit_news, 8*sizeof(T), cudaMemcpyDeviceToHost);
        if (sqpTimecheck()){ break; }


        line_search_step = 0;
        min_merit = h_merit_initial;
        for(int i = 0; i < 8; i++){
        //     std::cout << h_merit_news[i] << (i == 7 ? "\n" : " ");
            ///TODO: reduction ratio
            if(h_merit_news[i] < min_merit){
                min_merit = h_merit_news[i];
                line_search_step = i;
            }
        }
#ifdef SQP_DEBUG
        {
            uint32_t dzn = (state_size+control_size)*knot_points - control_size;
            std::vector<T> hdz(dzn); cudaMemcpy(hdz.data(), d_dz, dzn*sizeof(T), cudaMemcpyDeviceToHost);
            double nn=0; for(auto v:hdz) nn+=(double)v*(double)v;
            printf("[SQP_DEBUG] iter=%u ||dz||=%.4e rho=%.3e merit_init=%.6e min_merit=%.6e step=%d\n",
                   sqp_iter, sqrt(nn), (double)rho, (double)h_merit_initial, (double)min_merit, line_search_step);
        }
#endif


        if(min_merit == h_merit_initial){
            // line search failure
            drho = max(drho*rho_factor, rho_factor);
            rho = max(rho*drho, rho_min);
            sqp_iter++;
            if(rho > rho_max){
                sqp_time_exit = 0;
                rho = rho_reset;
                break; 
            }
            continue;
        }
        // std::cout << "line search accepted\n";
        alphafinal = -1.0 / (1 << line_search_step);        // alpha sign

        drho = min(drho/rho_factor, 1/rho_factor);
        rho = max(rho*drho, rho_min);
        

#if USE_DOUBLES
        cublasDaxpy(
            handle, 
            DZ_SIZE_BYTES / sizeof(T),
            &alphafinal,
            d_dz, 1,
            d_xu, 1
        );
#else
        cublasSaxpy(
            handle, 
            DZ_SIZE_BYTES / sizeof(T),
            &alphafinal,
            d_dz, 1,
            d_xu, 1
        );
#endif

        gpuErrchk(cudaPeekAtLastError());
        // if success increment after update
        sqp_iter++;

        if (sqpTimecheck()){ break; }


        delta_merit_iter = h_merit_initial - min_merit;
        delta_merit_total += delta_merit_iter;
        

        h_merit_initial = min_merit;
    
    }
    
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#ifndef MPCGPU_CORRECTNESS
    clock_gettime(CLOCK_MONOTONIC, &sqp_solve_end);
#endif

    // Ownership stays with the reusable workspace; a local fallback is RAII.
#ifdef MPCGPU_CORRECTNESS
    double sqp_solve_time = 0;
#else
    double sqp_solve_time = time_delta_us_timespec(sqp_solve_start, sqp_solve_end);
#endif

    return std::make_tuple(linsys_iter_vec, linsys_time_vec, sqp_solve_time, sqp_iter, sqp_time_exit, linsys_exit_vec);
}
