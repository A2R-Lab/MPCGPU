#include <glass.cuh>
#include <constants.cuh>
#include <gpuassert.cuh>
#include <interface.cuh>
#include <types.cuh>
#include <cmath>
#include <residual.cuh>
#include <linsys.cuh>


template <typename T>
__global__
void compute_gamma_kernel(T * d_gamma, T * d_g, T * d_A, T *d_x, T* d_lambda, T*d_z, float rho, float sigma){

	__shared__ T s_zdiff[NC];
	__shared__ T s_Atz[NX];

	/* z_diff = rho * z - lambda */
	glass::axpby<T>(NC, rho, d_z, -1, d_lambda, s_zdiff);

	/* Atx = A.T * z_diff */
	glass::gemv<T, true>(NC, NX, 1, d_A, s_zdiff, s_Atz);

	/* gamma = -g + sigma * x */
	glass::axpby<T>(NX, -1, d_g, sigma, d_x, d_gamma);

	/* gamma = Atz + gamma */
	glass::axpy<T>(NX, 1, s_Atz, d_gamma);

}

template <typename T>
void compute_gamma(T * d_gamma, T * d_g, T * d_A, T *d_x, T* d_lambda, T*d_z,  float rho, float sigma){
	/* gamma = -g + sigma * x + A.T ( rho *z - lambda )*/
	
	/* launch it*/

	void *compute_gamma_kernel_ptr = (void *) compute_gamma_kernel<T>;

	void *kernelArgs[] = {
		(void *)&d_gamma, 
		(void *)&d_g, 
		(void *)&d_A,
		(void *)&d_x, 
		(void *)&d_lambda, 
		(void *)&d_z, 
		(void *)&rho, 
		(void *)&sigma
	};

	gpuErrchk(cudaLaunchCooperativeKernel(compute_gamma_kernel_ptr, 1, NUM_THREADS, kernelArgs, 0));    
    gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
}


template <typename T>
__global__
void update_z_kernel(T *d_A, T *d_x, T *d_lambda, T *d_z, T *d_Ax , T* l, T* u, float rho ){

	/* Ax = A * x */
	glass::gemv<T, false>(NC, NX, 1, d_A, d_x, d_Ax);

	/* z = Ax + 1/rho * lambda */
	glass::axpby<T>(NC, 1, d_Ax, 1/rho, d_lambda, d_z);

	/* z = clip(z) */
	glass::clip(NC, d_z, l, u);

}


template <typename T>
void update_z(T *d_A, T *d_x, T *d_lambda, T *d_z,  T* d_l, T* d_u, float rho){
	/* z = clip( Ax + 1/rho * lambda )*/

	/* launch kernel*/

	/* Allocate for  Ax*/
	T * d_Ax;
	gpuErrchk(cudaMalloc(&d_Ax, NC * sizeof(T)));

	void *update_z_kernel_ptr = (void *) update_z_kernel<T>;

	void *kernelArgs[] = {
		(void *)&d_A,
		(void *)&d_x, 
		(void *)&d_lambda, 
		(void *)&d_z,  
		(void *)&d_Ax, 
		(void *)&d_l,
		(void *)&d_u,
		(void *)&rho
	};

	gpuErrchk(cudaLaunchCooperativeKernel(update_z_kernel_ptr, 1, NUM_THREADS, kernelArgs, 0));    
    gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	gpuErrchk(cudaFree(d_Ax));
}


template <typename T>
__global__
void update_lambda_kernel(T * d_A, T * d_x, T * d_lambda, T * d_z, T * d_Ax, T * d_Axz, float rho){

	/* Ax = A * x*/
	glass::gemv<T, false>(NC, NX, 1, d_A, d_x, d_Ax);

	/* Axz = Ax - z*/
	glass::axpby<T>(NC, 1, d_Ax, -1, d_z, d_Axz);

	/* lambda = lambda + rho * Axz */
	glass::axpy<T>(NC, rho, d_Axz, d_lambda);
}

template <typename T>
void update_lambda(T * d_A, T * d_x, T * d_lambda, T * d_z, float rho){
	/* lambda = lambda + rho * (A * x - z )*/

	// printf("Rho: %f\n\n", rho);

	T *d_Ax, *d_Axz;
	gpuErrchk(cudaMalloc(&d_Ax, NC * sizeof(T)));
	gpuErrchk(cudaMalloc(&d_Axz, NC * sizeof(T)));

	/* launch */
	void *update_lambda_kernel_ptr = (void *) update_lambda_kernel<T>;

	void *kernelArgs[] = {
		(void *)&d_A,
		(void *)&d_x, 
		(void *)&d_lambda, 
		(void *)&d_z,  
		(void *)&d_Ax,
		(void *)&d_Axz,
		(void *)&rho
	};

	gpuErrchk(cudaLaunchCooperativeKernel(update_lambda_kernel_ptr, 1, NUM_THREADS, kernelArgs, 0));    
    gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	gpuErrchk(cudaFree(d_Ax));
	gpuErrchk(cudaFree(d_Axz));
}


template <typename T>
__global__
void update_z_lambda_kernel(T *d_A, T *d_x, T *d_lambda, T *d_z, T* l, T* u, float rho ){
	__shared__ T s_Ax[NC];
	__shared__ T s_Axz[NC];

	/* Ax = A * x */
	glass::gemv<T, false>(NC, NX, 1, d_A, d_x, s_Ax);

	/* z = Ax + 1/rho * lambda */
	glass::axpby<T>(NC, 1, s_Ax, 1/rho, d_lambda, d_z);

	/* z = clip(z) */
	glass::clip(NC, d_z, l, u);

	/* Axz = Ax - z*/
	glass::axpby<T>(NC, 1, s_Ax, -1, d_z, s_Axz);

	/* lambda = lambda + rho * Axz */
	glass::axpy<T>(NC, rho, s_Axz, d_lambda);

}


template <typename T>
void update_z_lambda(T *d_A, T *d_x, T *d_lambda, T *d_z,  T* d_l, T* d_u, float rho){
	/* z = clip( Ax + 1/rho * lambda )*/

	/* launch kernel*/

	void *update_z_lambda_kernel_ptr = (void *) update_z_lambda_kernel<T>;

	void *kernelArgs[] = {
		(void *)&d_A,
		(void *)&d_x, 
		(void *)&d_lambda, 
		(void *)&d_z,   
		(void *)&d_l,
		(void *)&d_u,
		(void *)&rho
	};

	gpuErrchk(cudaLaunchCooperativeKernel(update_z_lambda_kernel_ptr, 1, NUM_THREADS, kernelArgs, 0));    
    gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
}


template <typename T>
void admm_iter(qp<T> *prob, T *d_x, T *d_lambda, T *d_z, float rho, float sigma){
	T *d_Pinv, * d_Sn, * d_Sbd;
    gpuErrchk(cudaMalloc(&d_Pinv, 3*STATE_SIZE*STATE_SIZE*KNOT_POINTS*sizeof(T)));
	gpuErrchk(cudaMalloc(&d_Sn, NX * NX * sizeof(T)));
	gpuErrchk(cudaMalloc(&d_Sbd, 3*STATE_SIZE*STATE_SIZE*KNOT_POINTS*sizeof(T)));


	/* form_schur */
	form_schur(d_Sn, prob->d_H, prob->d_A, prob->d_Anorm, rho, sigma);
	cudaDeviceSynchronize();

	/* convert to custom bd form */
	convert_to_bd(d_Sn, d_Sbd);

	/* TODO: form_precon from schur */
	form_ss(d_Pinv, d_Sbd);

	/* Allocate memory for gamma, */
	T *d_gamma;
	gpuErrchk(cudaMalloc(&d_gamma, NX * sizeof(T)));

	/*compute gamma*/
	compute_gamma<T>(d_gamma, prob->d_g, prob->d_A, d_x, d_lambda, d_z, rho, sigma);

	/*call pcg*/
	solve_pcg<T>(d_Sbd, d_Pinv, d_gamma, d_x);

	/*update z and lambda*/
	update_z_lambda<T>(prob->d_A, d_x, d_lambda, d_z, prob->d_l, prob->d_u, rho);

	gpuErrchk(cudaFree(d_Pinv));
	gpuErrchk(cudaFree(d_Sn));
	gpuErrchk(cudaFree(d_Sbd));
	gpuErrchk(cudaFree(d_gamma));
}