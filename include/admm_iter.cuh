#include <glass.cuh>
#define NUM_THREADS 64
#define STATE_SIZE 10
#define CONTROL_SIZE 2
#define KNOT_POINTS 10
#define MSIZE (STATE_SIZE+CONTROL_SIZE)*KNOT_POINTS
#define SSpCS (STATE_SIZE+CONTROL_SIZE)

template <typename T>
compute_gamma(T * d_gamma, T * d_g, T * d_A, t *d_x, T* d_lambda, float rho, float sigma){
	/* gamma = g + sigma * x + A.T ( rho *x - lambda )*/
	
	/* launch it*/

	update_gamma_skernel<T><<<nx, NUM_THREADS>>>(d_gamma, d_g, d_A, d_x, d_lambda, d_z, rho, sigma);

}


template <typename T>
compute_gamma_kernel(){
	/* x_diff = rho * x - lambda */
	glass::axpby<T>(nx, rho, d_x, -1, lambda, d_xdiff);

	/* Atx = A.T * x_diff */
	glass::gemv<T, true>(nx, nx, 1, d_A, d_xdiff, d_Atx);

	/* gamma = g + sigma *x */
	glass::axpby<T>(nx, 1, d_g, sigma, x, d_gamma);

	/* gamma = Atx + gamma */
	glass::axpy<T>(nx, 1, d_Atx, d_gamma);
}


template <typename T>
update_z(T *d_A, T *d_x, T *d_lambda, T *d_z, float rho, T* l, T* u){
	/* z = clip( Ax + 1/rho * lambda )*/

	/* launch kernel*/
	
	update_z_kernel<T><<<nx, NUM_THREADS>>>(d_A, d_x, d_lambda, d_z, rho, l, u);

}

template <typename T>
update_z_kernel(T *d_A, T *d_x, T *d_lambda, T *d_z,  T * l, T * u, float rho, , T* l, T* u){
	/* Ax = A * x */
	glass::gemv<T, false>(nx, nx, 1, d_A, d_x, d_Ax);

	/* z = Ax + 1/rho * lambda */
	glass::axpby<T>(nx, 1, d_Ax, 1/rho, lambda, d_z);

	/* z = clip(z) */
	glass::clip(nx, d_z, l, u);

}

template <typename T>
update_lambda(T * d_A, T * d_x, T * d_lambda, T * d_z, float rho){
	/* lambda = lambda + rho * (A * x - z )*/

	/* launch */
	update_lambda_kernel<T><<<nx, NUM_THREADS>>>(d_A, d_x, d_lambda, d_z, rho);
}

template <typename T>
update_lambda_kernel(T * d_A, T * d_x, T * d_lambda, T * d_z, float rho){
	/* Ax = A * x*/
	glass::gemv<T, false>(nx, nx, 1, d_A, d_x, d_Ax);

	/* Axz = Ax - z*/
	glass::axpby<T>(nx, 1, d_Ax, -1, d_z, d_Axz);

	/* lambda = lambda + rho * Axz */
	glass::axpy<T>(nx, rho, d_Axz, d_lambda);
}

template <typename T>
solve_pcg(T * d_S, T * d_Pinv, T * d_x){
	pcg_config config;
	config.empty_pinv = 0;
    
    /*   PCG vars   */
    T  *d_r, *d_p, *d_v_temp, *d_eta_new_temp;// *d_r_tilde, *d_upsilon;
    gpuErrchk(cudaMalloc(&d_r, state_size*knot_points*sizeof(T)));
    gpuErrchk(cudaMalloc(&d_p, state_size*knot_points*sizeof(T)));
    gpuErrchk(cudaMalloc(&d_v_temp, knot_points*sizeof(T)));
    gpuErrchk(cudaMalloc(&d_eta_new_temp, knot_points*sizeof(T)));
    
    
    
    void *pcg_kernel = (void *) pcg<T, STATE_SIZE, KNOT_POINTS>;
    uint32_t pcg_iters;
    uint32_t *d_pcg_iters;
    gpuErrchk(cudaMalloc(&d_pcg_iters, sizeof(uint32_t)));
    bool pcg_exit;
    bool *d_pcg_exit;
    gpuErrchk(cudaMalloc(&d_pcg_exit, sizeof(bool)));
    
    void *pcgKernelArgs[] = {
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
        (void *)&config.pcg_max_iter,
        (void *)&config.pcg_exit_tol
    };
    size_t ppcg_kernel_smem_size = pcgSharedMemSize<T>(state_size, knot_points);


    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

	gpuErrchk(cudaLaunchCooperativeKernel(pcg_kernel, knot_points, NUM_THREADS, pcgKernelArgs, ppcg_kernel_smem_size));    
	gpuErrchk(cudaMemcpy(&pcg_iters, d_pcg_iters, sizeof(uint32_t), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(&pcg_exit, d_pcg_exit, sizeof(bool), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaPeekAtLastError());
}

template <typename T>
admm_iter(qp *prob, T *d_S, T *d_Pinv, T *d_x, T *d_lambda, T *d_z, float rho, float sigma){

	/*Allocate memory for gamma, */

	/*compute gamma*/
	compute_gamma(d_gamma, prob->d_g, prob->d_A, d_x, d_lambda, rho, sigma);

	/*call pcg*/
	solve_pcg(d_S, d_Pinv, d_x);

	/*update z*/
	update_z(d_A, d_x, d_lambda, d_z, rho, prob->l, prob->u);

	/*update lambda*/
	update_lambda(d_A, d_x, d_lambda, d_z, rho);
}