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

	QDLDL_factor(An,Ap,Ai,Ax,Lp,Li,Lx,D,Dinv,Lnz,etree,bwork,iwork,fwork);

	for(i=0;i < Ln; i++) x[i] = b[i];

	QDLDL_solve(Ln,Lp,Li,Lx,Dinv,x);
}


template <typename T>
auto sqpSolveQdldl(uint32_t state_size, uint32_t control_size, uint32_t knot_points, float timestep, T *d_eePos_traj, T *d_lambda, T *d_xu, void *d_dynMem_const, T &rho, T rho_reset){
    
    // data storage
    std::vector<int> linsys_iter_vec;
    std::vector<bool> linsys_exit_vec;
    std::vector<double> linsys_time_vec;
    bool sqp_time_exit = 1;     // for data recording, not a flag
    


    // sqp timing
    struct timespec sqp_solve_start, sqp_solve_end;
    gpuErrchk(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &sqp_solve_start);


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


    // streams n cublas init
    cudaStream_t streams[num_alphas];
    for(uint32_t str = 0; str < num_alphas; str++){
        cudaStreamCreate(&streams[str]);
    }
    gpuErrchk(cudaPeekAtLastError());

    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { printf ("CUBLAS initialization failed\n"); exit(13); }
    gpuErrchk(cudaPeekAtLastError());


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

    


    gpuErrchk(cudaMalloc(&d_G_dense,  KKT_G_DENSE_SIZE_BYTES));
    gpuErrchk(cudaMalloc(&d_C_dense,  KKT_C_DENSE_SIZE_BYTES));
    gpuErrchk(cudaMalloc(&d_g,        KKT_g_SIZE_BYTES));
    gpuErrchk(cudaMalloc(&d_c,        KKT_c_SIZE_BYTES));
    d_Ginv_dense = d_G_dense;

    gpuErrchk(cudaMalloc(&d_S, 3*states_sq*knot_points*sizeof(T)));
    gpuErrchk(cudaMalloc(&d_gamma, state_size*knot_points*sizeof(T)));
    gpuErrchk(cudaPeekAtLastError());

    
    gpuErrchk(cudaMalloc(&d_dz,       DZ_SIZE_BYTES));
    gpuErrchk(cudaMalloc(&d_xs,       state_size*sizeof(T)));
    gpuErrchk(cudaMemcpy(d_xs, d_xu,  state_size*sizeof(T), cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMalloc(&d_merit_news, 8*sizeof(T)));
    gpuErrchk(cudaMalloc(&d_merit_temp, 8*knot_points*sizeof(T)));
    // linsys iterates

    gpuErrchk(cudaMalloc(&d_merit_initial, sizeof(T)));
    gpuErrchk(cudaMemset(d_merit_initial, 0, sizeof(T)));
    



    const int nnz = (knot_points-1)*states_sq + knot_points*(((state_size+1)*state_size)/2);
    
    QDLDL_float h_lambda[state_size*knot_points];
    QDLDL_float h_gamma[state_size*knot_points];
    QDLDL_int h_col_ptr[state_size*knot_points+1];
    QDLDL_int h_row_ind[nnz];
    QDLDL_float h_val[nnz];
    
    QDLDL_int *d_row_ind, *d_col_ptr;
    QDLDL_float *d_val, *d_lambda_double;
    gpuErrchk(cudaMalloc(&d_col_ptr, (state_size*knot_points+1)*sizeof(QDLDL_int)));
    gpuErrchk(cudaMalloc(&d_row_ind, nnz*sizeof(QDLDL_int)));
	gpuErrchk(cudaMalloc(&d_val, nnz*sizeof(QDLDL_float)));
	gpuErrchk(cudaMalloc(&d_lambda_double, (state_size*knot_points)*sizeof(QDLDL_float)));
    
    // fill col ptr and row ind, these won't change 
    prep_csr<<<knot_points, 64>>>(state_size, knot_points, d_col_ptr, d_row_ind);
    gpuErrchk(cudaMemcpy(h_col_ptr, d_col_ptr, (state_size*knot_points+1)*sizeof(QDLDL_int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_row_ind, d_row_ind, (nnz)*sizeof(QDLDL_int), cudaMemcpyDeviceToHost));

    
    const QDLDL_int An = state_size*knot_points;

    // Q things
    QDLDL_int  sumLnz;
    QDLDL_int *etree;
	QDLDL_int *Lnz;
    etree = (QDLDL_int*)malloc(sizeof(QDLDL_int)*An);
	Lnz   = (QDLDL_int*)malloc(sizeof(QDLDL_int)*An);
    
    QDLDL_int *Lp;
	QDLDL_float *D;
	QDLDL_float *Dinv;
    Lp    = (QDLDL_int*)malloc(sizeof(QDLDL_int)*(An+1));
	D     = (QDLDL_float*)malloc(sizeof(QDLDL_float)*An);
	Dinv  = (QDLDL_float*)malloc(sizeof(QDLDL_float)*An);

    //working data for factorisation
	QDLDL_int   *iwork;
	QDLDL_bool  *bwork;
	QDLDL_float *fwork;
    iwork = (QDLDL_int*)malloc(sizeof(QDLDL_int)*(3*An));
	bwork = (QDLDL_bool*)malloc(sizeof(QDLDL_bool)*An);
	fwork = (QDLDL_float*)malloc(sizeof(QDLDL_float)*An);

    sumLnz = QDLDL_etree(An,h_col_ptr,h_row_ind,iwork,Lnz,etree);
    
    QDLDL_int *Li;
	QDLDL_float *Lx;
    Li    = (QDLDL_int*)malloc(sizeof(QDLDL_int)*sumLnz);
	Lx    = (QDLDL_float*)malloc(sizeof(QDLDL_float)*sumLnz);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#if TIME_LINSYS == 1
    struct timespec linsys_start, linsys_end;
    double linsys_time;
#endif
#if CONST_UPDATE_FREQ
    struct timespec sqp_cur;
    auto sqpTimecheck = [&]() {
        clock_gettime(CLOCK_MONOTONIC, &sqp_cur);
        return time_delta_us_timespec(sqp_solve_start,sqp_cur) > SQP_MAX_TIME_US;
    };
#else
    auto sqpTimecheck = [&]() { return false; };
#endif


    ///TODO: atomic race conditions here aren't fixed but don't seem to be problematic
    compute_merit<T><<<knot_points, MERIT_THREADS, merit_smem_size>>>(
        state_size, control_size, knot_points,
        d_xu, 
        d_eePos_traj, 
        static_cast<T>(10), 
        timestep, 
        d_dynMem_const, 
        d_merit_initial
    );
    gpuErrchk(cudaMemcpyAsync(&h_merit_initial, d_merit_initial, sizeof(T), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaPeekAtLastError());

    // gpuErrchk(cudaDeviceSynchronize());
    // std::cout << "initial merit " << h_merit_initial << std::endl;
    // exit(0);

    //
    //      SQP LOOP
    //
    for(uint32_t sqpiter = 0; sqpiter < SQP_MAX_ITER; sqpiter++){
        
        generate_kkt_submatrices<T><<<knot_points, KKT_THREADS, 2 * get_kkt_smem_size<T>(state_size, control_size)>>>(
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
            d_xu
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
                (void *)&d_merit_temp
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
    clock_gettime(CLOCK_MONOTONIC, &sqp_solve_end);

    cublasDestroy(handle);

    for(uint32_t st=0; st < num_alphas; st++){
        gpuErrchk(cudaStreamDestroy(streams[st]));
    }




    gpuErrchk(cudaFree(d_merit_initial));
    gpuErrchk(cudaFree(d_merit_news));
    gpuErrchk(cudaFree(d_merit_temp));
    gpuErrchk(cudaFree(d_G_dense));
    gpuErrchk(cudaFree(d_C_dense));
    gpuErrchk(cudaFree(d_g));
    gpuErrchk(cudaFree(d_c));
    gpuErrchk(cudaFree(d_S));
    gpuErrchk(cudaFree(d_gamma));
    gpuErrchk(cudaFree(d_dz));
    gpuErrchk(cudaFree(d_xs));
    gpuErrchk(cudaFree(d_col_ptr));
    gpuErrchk(cudaFree(d_row_ind));
    gpuErrchk(cudaFree(d_val));
    gpuErrchk(cudaFree(d_lambda_double));
	free(etree);
	free(Lnz);
    free(Lp);
	free(D);
	free(Dinv);
	free(iwork);
	free(bwork);
	free(fwork);
	free(Li);
	free(Lx);

    double sqp_solve_time = time_delta_us_timespec(sqp_solve_start, sqp_solve_end);

    return std::make_tuple(linsys_iter_vec, linsys_time_vec, sqp_solve_time, sqp_iter, sqp_time_exit, linsys_exit_vec);
}
