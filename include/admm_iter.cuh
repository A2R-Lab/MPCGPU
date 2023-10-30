#include <glass.cuh>
#define NUM_THREADS 64
#define KNOT_POINTS 10
#define MSIZE (STATE_SIZE+CONTROL_SIZE)*KNOT_POINTS
#define SSpCS (STATE_SIZE+CONTROL_SIZE)
#define NX 120
#define NC 120
#define STATE_SIZE NX/KNOT_POINTS

template <typename T>
compute_gamma(T * d_gamma, T * d_g, T * d_A, t *d_x, T* d_lambda, T*d_z,  float rho, float sigma){
	/* gamma = -g + sigma * x + A.T ( rho *z - lambda )*/
	
	/* launch it*/

	T *d_zdiff, T *d_Atz;
	gpuErrchk(cudaMalloc(d_zdiff, NC * sizeof(T)));
	gpuErrchk(cudaMalloc(d_Atz, NX * sizeof(T)));

	void *compute_gamma_kernel = (void *) compute_gamma_kernel<T>;

	void *kernelArgs[] = {
		(void *)&d_gamma, 
		(void *)&d_g, 
		(void *)&d_A,
		(void *)&d_x, 
		(void *)&d_lambda, 
		(void *)&d_z, 
		(void *)&d_zdiff, 
		(void *)&d_Atz, 
		(void *)&rho, 
		(void *)&sigma
	}

	gpuErrchk(cudaLaunchCooperativeKernel(compute_gamma_kernel, knot_points, NUM_THREADS, kernelArgs, 0));    
    gpuErrchk(cudaPeekAtLastError());
}


template <typename T>
compute_gamma_kernel(T * d_gamma, T * d_g, T * d_A, t *d_x, T* d_lambda, T*d_z, T *d_zdiff, T *d_Atz, float rho, float sigma){

	/* z_diff = rho * z - lambda */
	glass::axpby<T>(NC, rho, d_z, -1, d_lambda, d_zdiff);

	/* Atx = A.T * z_diff */
	glass::gemv<T, true>(NC, NX, 1, d_A, d_zdiff, d_Atz);

	/* gamma = -g + sigma * x */
	glass::axpby<T>(NX, -1, d_g, sigma, x, d_gamma);

	/* gamma = Atz + gamma */
	glass::axpy<T>(NX, 1, d_Atz, d_gamma);
}


template <typename T>
update_z(T *d_A, T *d_x, T *d_lambda, T *d_z,  T* l, T* u, float rho){
	/* z = clip( Ax + 1/rho * lambda )*/

	/* launch kernel*/

	/* Allocate for  Ax*/
	T * d_Ax;
	gpuErrchk(cudaMalloc(d_Ax, NC * sizeof(T)));

	void *update_z_kernel = (void *) compute_z_kernel<T>;

	void *kernelArgs[] = {
		(void *)&d_A,
		(void *)&d_x, 
		(void *)&d_lambda, 
		(void *)&d_z,  
		(void *)&d_Ax, 
		(void *)&d_l,
		(void *)&d_u,
		(void *)&rho
	}

	gpuErrchk(cudaLaunchCooperativeKernel(update_z_kernel, knot_points, NUM_THREADS, kernelArgs, 0));    
    gpuErrchk(cudaPeekAtLastError());
}

template <typename T>
update_z_kernel(T *d_A, T *d_x, T *d_lambda, T *d_z, d_Ax , T* l, T* u float rho ){
	/* Ax = A * x */
	glass::gemv<T, false>(NC, NX, 1, d_A, d_x, d_Ax);

	/* z = Ax + 1/rho * lambda */
	glass::axpby<T>(NC, 1, d_Ax, 1/rho, lambda, d_z);

	/* z = clip(z) */
	glass::clip(NC, d_z, l, u);

}

template <typename T>
update_lambda(T * d_A, T * d_x, T * d_lambda, T * d_z, float rho){
	/* lambda = lambda + rho * (A * x - z )*/

	T * d_Ax, T d_Axz;
	gpuErrchk(cudaMalloc(d_Ax, NC * sizeof(T)));
	gpuErrchk(cudaMalloc(d_Axz, NC * sizeof(T)));

	/* launch */
	void *update_lambda_kernel = (void *) compute_lambda_kernel<T>;

	void *kernelArgs[] = {
		(void *)&d_A,
		(void *)&d_x, 
		(void *)&d_lambda, 
		(void *)&d_z,  
		(void *)&d_Ax,
		(void *)&d_Axz,
		(void *)&rho
	}

	gpuErrchk(cudaLaunchCooperativeKernel(update_lambda_kernel, knot_points, NUM_THREADS, kernelArgs, 0));    
    gpuErrchk(cudaPeekAtLastError());
}

template <typename T>
update_lambda_kernel(T * d_A, T * d_x, T * d_lambda, T * d_z, T * d_Ax, T * d_Axz, float rho){
	/* Ax = A * x*/
	glass::gemv<T, false>(NC, NX, 1, d_A, d_x, d_Ax);

	/* Axz = Ax - z*/
	glass::axpby<T>(NC, 1, d_Ax, -1, d_z, d_Axz);

	/* lambda = lambda + rho * Axz */
	glass::axpy<T>(NC, rho, d_Axz, d_lambda);
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
	gpuErrchk(cudaPeekAtLastError());
}

template <typename T>
admm_iter(qp *prob, T *d_S, T *d_Pinv, T *d_x, T *d_lambda, T *d_z, float rho, float sigma){

	/*Allocate memory for gamma, */
	T *d_gamma;
	gpuErrchk(cudaMalloc(d_gamma, NX * sizeof(T)));

	/*compute gamma*/
	compute_gamma<T(d_gamma, prob->d_g, prob->d_A, d_x, d_lambda, d_z, rho, sigma);

	/*call pcg*/
	solve_pcg<T>(d_S, d_Pinv, d_x);

	/*update z*/
	update_z<T>(d_A, d_x, d_lambda, d_z, prob->l, prob->u, rho);

	/*update lambda*/
	update_lambda<T>(d_A, d_x, d_lambda, d_z, rho);
}