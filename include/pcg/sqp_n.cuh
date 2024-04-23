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
#include "linsys_setup.cuh"
#include "common/kkt.cuh"
#include "common/dz.cuh"
#include "merit.cuh"
#include "gpu_pcg.cuh"
#include "settings.cuh"

template <typename T>
auto sqpSolvePcg(const uint32_t solve_count, const uint32_t state_size, const uint32_t control_size, const uint32_t knot_points, float timestep, T *d_eePos_traj, T *d_lambda, T *d_xu, void *d_dynMem_const, pcg_config<T>& config, T &rho, T rho_reset){
    

    // data storage
    std::vector<std::vector<int>> pcg_iter_vec;
    std::vector<std::vector<bool>> pcg_exit_vec;
    std::vector<std::vector<double>> linsys_time_vec;
    bool sqp_time_exit = 1;     // for data recording, not a flag, not sure what this does?


    // sqp timing
    struct timespec sqp_solve_start, sqp_solve_end;
    gpuErrchk(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &sqp_solve_start);


    const uint32_t states_sq = state_size*state_size;
    const uint32_t states_p_controls = state_size * control_size;
    const uint32_t controls_sq = control_size * control_size;
    const uint32_t states_s_controls = state_size + control_size;
    const uint32_t traj_len = (state_size+control_size)*knot_points-control_size;
    const uint32_t KKT_G_DENSE_SIZE_BYTES = static_cast<uint32_t>(((states_sq+controls_sq)*knot_points-controls_sq)*sizeof(T));
    const uint32_t KKT_C_DENSE_SIZE_BYTES = static_cast<uint32_t>((states_sq+states_p_controls)*(knot_points-1)*sizeof(T));
    const uint32_t KKT_g_SIZE_BYTES       = static_cast<uint32_t>(((state_size+control_size)*knot_points-control_size)*sizeof(T));
    const uint32_t KKT_c_SIZE_BYTES       =   static_cast<uint32_t>((state_size*knot_points)*sizeof(T));     
    const uint32_t DZ_SIZE_BYTES          =   static_cast<uint32_t>((states_s_controls*knot_points-control_size)*sizeof(T));


    // line search things
    const float mu = 10.0f;
    const uint32_t num_alphas = 8;
    T h_merit_news[solve_count][num_alphas];
    void *ls_merit_kernel = (void *) ls_gato_compute_merit<T>;
    const size_t merit_smem_size = get_merit_smem_size<T>(state_size, control_size);
    T h_merit_initial[solve_count], min_merit[solve_count];
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

    


    gpuErrchk(cudaMalloc(&d_G_dense,  KKT_G_DENSE_SIZE_BYTES * solve_count));
    gpuErrchk(cudaMalloc(&d_C_dense,  KKT_C_DENSE_SIZE_BYTES * solve_count));
    gpuErrchk(cudaMalloc(&d_g,        KKT_g_SIZE_BYTES  * solve_count));
    gpuErrchk(cudaMalloc(&d_c,        KKT_c_SIZE_BYTES * solve_count));
    d_Ginv_dense = d_G_dense;

    gpuErrchk(cudaMalloc(&d_S, 3*states_sq*knot_points*sizeof(T) * solve_count));
    gpuErrchk(cudaMalloc(&d_gamma, state_size*knot_points*sizeof(T) * solve_count));
    gpuErrchk(cudaPeekAtLastError());

    
    gpuErrchk(cudaMalloc(&d_dz,       DZ_SIZE_BYTES * solve_count));
    gpuErrchk(cudaMalloc(&d_xs,       state_size*sizeof(T) * solve_count));
    gpuErrchk(cudaMemcpy(d_xs, d_xu,  state_size*sizeof(T) * solve_count, cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMalloc(&d_merit_news, 8*sizeof(T) * solve_count));
    gpuErrchk(cudaMalloc(&d_merit_temp, 8*knot_points*sizeof(T) * solve_count));
    // pcg iterates

    gpuErrchk(cudaMalloc(&d_merit_initial, sizeof(T) * solve_count));
    gpuErrchk(cudaMemset(d_merit_initial, 0, sizeof(T) * solve_count));
    

    // pcg things
    T *d_Pinv;
    gpuErrchk(cudaMalloc(&d_Pinv, 3*states_sq*knot_points*sizeof(T) * solve_count));
    
    /*   PCG vars   */
    T  *d_r, *d_p, *d_v_temp, *d_eta_new_temp;// *d_r_tilde, *d_upsilon;
    gpuErrchk(cudaMalloc(&d_r, state_size*knot_points*sizeof(T) * solve_count));
    gpuErrchk(cudaMalloc(&d_p, state_size*knot_points*sizeof(T) * solve_count));
    gpuErrchk(cudaMalloc(&d_v_temp, knot_points*sizeof(T) * solve_count));
    gpuErrchk(cudaMalloc(&d_eta_new_temp, knot_points*sizeof(T) * solve_count));
    
    
    
    void *pcg_kernel = (void *) pcg<T, STATE_SIZE, KNOT_POINTS>;
    uint32_t pcg_iters;
    uint32_t *d_pcg_iters;
    gpuErrchk(cudaMalloc(&d_pcg_iters, sizeof(uint32_t));
    bool pcg_exit[solve_count];
    bool *d_pcg_exit;
    gpuErrchk(cudaMalloc(&d_pcg_exit, sizeof(bool)));
   
    size_t ppcg_kernel_smem_size = pcgSharedMemSize<T>(state_size, knot_points);


    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

#if TIME_LINSYS
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
    for (uint32_t prob = 0; prob < solve_count; prob++){
        compute_merit<T><<<knot_points, MERIT_THREADS, merit_smem_size>>>(
            state_size, control_size, knot_points,
            d_xu, // d_xu + traj_len * prob, 
            d_eePos_traj, // d_eePos_traj + 6 * knot_points * prob, 
            static_cast<T>(10), 
            timestep, 
            d_dynMem_const, 
            d_merit_initial + prob
        );
        gpuErrchk(cudaMemcpyAsync(&h_merit_initial[prob], d_merit_initial + prob, sizeof(T), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaPeekAtLastError());
    }

    //
    //      SQP LOOP
    //
    for(uint32_t sqpiter = 0; sqpiter < SQP_MAX_ITER; sqpiter++){
        std::vector<int> pcg_iter_vec_temp;
        std::vector<bool> pcg_exit_vec_temp;
        std::vector<double> linsys_time_vec_temp;

        for (uint32_t prob = 0; prob < solve_count; prob++){
            generate_kkt_submatrices<T><<<knot_points, KKT_THREADS, 2 * get_kkt_smem_size<T>(state_size, control_size)>>>(
                state_size,
                control_size,
                knot_points,
                d_G_dense, // d_G_dense + KKT_G_DENSE_SIZE_BYTES / sizeof(T) * prob,
                d_C_dense, // d_C_dense + KKT_C_DENSE_SIZE_BYTES / sizeof(T) * prob,
                d_g, // d_g + KKT_g_SIZE_BYTES / sizeof(T) * prob,
                d_c, // d_c + KKT_c_SIZE_BYTES / sizeof(T) * prob,
                d_dynMem_const,
                timestep,
                d_eePos_traj, // d_eePos_traj + 6 * knot_points * prob,
                d_xs, // d_xs + state_size * prob,
                d_xu // d_xu + traj_len * prob
            );
            gpuErrchk(cudaPeekAtLastError());
            if (sqpTimecheck()){ break; }

            form_schur_system<T>(
                state_size, 
                control_size, 
                knot_points, 
                d_G_dense, // d_G_dense + KKT_G_DENSE_SIZE_BYTES / sizeof(T) * prob,
                d_C_dense, // d_C_dense + KKT_C_DENSE_SIZE_BYTES / sizeof(T) * prob,
                d_g, // d_g + KKT_g_SIZE_BYTES / sizeof(T) * prob,
                d_c, // d_c + KKT_c_SIZE_BYTES / sizeof(T) * prob,
                d_S, // d_S + 3*states_sq*knot_points * prob,
                d_Pinv, // d_Pinv + 3*states_sq*knot_points * prob,
                d_gamma, // d_gamma + state_size*knot_points * prob,
                rho // not sure if instanced?
            );
            gpuErrchk(cudaPeekAtLastError());
            if (sqpTimecheck()){ break; }
            

        #if TIME_LINSYS    
            gpuErrchk(cudaDeviceSynchronize());
            if (sqpTimecheck()){ break; }
            clock_gettime(CLOCK_MONOTONIC,&linsys_start);
        #endif // #if TIME_LINSYS

            void *pcgKernelArgs[] = {
                (void *)&d_S, // &(d_S + 3*states_sq * knot_points * prob),
                (void *)&d_Pinv, // &(d_Pinv + 3*states_sq * knot_points * prob),
                (void *)&d_gamma, // &(d_gamma + state_size * knot_points * prob),
                (void *)&d_lambda, // &(d_lambda + states_size * knot_points * prob),
                (void *)&d_r, // &(d_r + state_size * knot_points * prob),
                (void *)&d_p, // &(d_p + state_size * knot_points * prob),
                (void *)&d_v_temp, // &(d_v_temp + knot_points * prob),
                (void *)&d_eta_new_temp, // &(d_eta_new_temp + knot_points * prob),
                (void *)&d_pcg_iters,
                (void *)&d_pcg_exit,
                (void *)&config.pcg_max_iter,
                (void *)&config.pcg_exit_tol
            };

            gpuErrchk(cudaLaunchCooperativeKernel(pcg_kernel, knot_points, PCG_NUM_THREADS, pcgKernelArgs, ppcg_kernel_smem_size));    
            gpuErrchk(cudaMemcpy(&pcg_iters, d_pcg_iters, sizeof(uint32_t), cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpy(&pcg_exit, d_pcg_exit, sizeof(bool), cudaMemcpyDeviceToHost));
            gpuErrchk(cudaPeekAtLastError());

        #if TIME_LINSYS
            gpuErrchk(cudaDeviceSynchronize());
            clock_gettime(CLOCK_MONOTONIC,&linsys_end);
            
            linsys_time = time_delta_us_timespec(linsys_start,linsys_end);
            linsys_time_vec_temp.push_back(linsys_time);
        #endif // #if TIME_LINSYS

            pcg_iter_vec_temp.push_back(pcg_iters);
            pcg_exit_vec_temp.push_back(pcg_exit);

            
            if (sqpTimecheck()){ break; }
            
            // recover dz
            compute_dz(
                state_size,
                control_size,
                knot_points,
                d_Ginv_dense, // d_Ginv_dense + KKT_G_DENSE_SIZE_BYTES / sizeof(T) * prob,
                d_C_dense,  // d_G_dense + KKT_G_DENSE_SIZE_BYTES / sizeof(T) * prob,
                d_g, // d_g + KKT_g_SIZE_BYTES / sizeof(T) * prob,
                d_lambda, // d_lambda + states_size * knot_points * prob,
                d_dz // d_dz + DZ_SIZE_BYTES / sizeof(T) * prob
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
            
            
            cudaMemcpy(h_merit_news + 8 * prob, d_merit_news + 8 * prob, 8*sizeof(T), cudaMemcpyDeviceToHost);
            if (sqpTimecheck()){ break; }
            gpuErrchk(cudaPeekAtLastError());


            line_search_step = 0;
            min_merit[prob] = h_merit_initial[prob];
            for(int i = 0; i < 8; i++){
            //     std::cout << h_merit_news[i] << (i == 7 ? "\n" : " ");
                ///TODO: reduction ratio
                if(h_merit_news[prob][i] < min_merit[prob]){
                    min_merit[prob] = h_merit_news[prob][i];
                    line_search_step = i;
                }
            }


            if(min_merit[prob] == h_merit_initial[prob]){
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


            delta_merit_iter = h_merit_initial[prob] - min_merit[prob];
            delta_merit_total += delta_merit_iter;
            

            h_merit_initial[prob] = min_merit[prob];
        }
        pcg_iter_vec.push_back(pcg_iter_vec_temp);
        pcg_exit_vec.push_back(pcg_exit_vec_temp);
        linsys_time_vec.push_back(linsys_time_vec_temp);
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
    gpuErrchk(cudaFree(d_pcg_iters));
    gpuErrchk(cudaFree(d_pcg_exit));
    gpuErrchk(cudaFree(d_Pinv));
    gpuErrchk(cudaFree(d_r));
    gpuErrchk(cudaFree(d_p));
    gpuErrchk(cudaFree(d_v_temp));
    gpuErrchk(cudaFree(d_eta_new_temp));



    double sqp_solve_time = time_delta_us_timespec(sqp_solve_start, sqp_solve_end);

    return std::make_tuple(pcg_iter_vec[0], linsys_time_vec[0], sqp_solve_time, sqp_iter, sqp_time_exit, pcg_exit_vec[0]);
}
