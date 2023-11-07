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
void compute_gamma_kernel(T * d_gamma, T * d_g, T * d_A, T *d_x, T* d_lambda, T*d_z, T *d_zdiff, T *d_Atz, float rho, float sigma){

	/* z_diff = rho * z - lambda */
	glass::axpby<T>(NC, rho, d_z, -1, d_lambda, d_zdiff);

	/* Atx = A.T * z_diff */
	glass::gemv<T, true>(NC, NX, 1, d_A, d_zdiff, d_Atz);

	/* gamma = -g + sigma * x */
	glass::axpby<T>(NX, -1, d_g, sigma, d_x, d_gamma);

	/* gamma = Atz + gamma */
	glass::axpy<T>(NX, 1, d_Atz, d_gamma);
}

template <typename T>
void compute_gamma(T * d_gamma, T * d_g, T * d_A, T *d_x, T* d_lambda, T*d_z,  float rho, float sigma){
	/* gamma = -g + sigma * x + A.T ( rho *z - lambda )*/
	
	/* launch it*/

	T *d_zdiff, *d_Atz;
	gpuErrchk(cudaMalloc(&d_zdiff, NC * sizeof(T)));
	gpuErrchk(cudaMalloc(&d_Atz, NX * sizeof(T)));

	void *compute_gamma_kernel_ptr = (void *) compute_gamma_kernel<T>;

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
	};

	gpuErrchk(cudaLaunchCooperativeKernel(compute_gamma_kernel_ptr, KNOT_POINTS, NUM_THREADS, kernelArgs, 0));    
    gpuErrchk(cudaPeekAtLastError());
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

	gpuErrchk(cudaLaunchCooperativeKernel(update_z_kernel_ptr, KNOT_POINTS, NUM_THREADS, kernelArgs, 0));    
    gpuErrchk(cudaPeekAtLastError());
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

	gpuErrchk(cudaLaunchCooperativeKernel(update_lambda_kernel_ptr, KNOT_POINTS, NUM_THREADS, kernelArgs, 0));    
    gpuErrchk(cudaPeekAtLastError());
}



template <typename T>
void admm_iter(qp<T> *prob, T *d_x, T *d_lambda, T *d_z, float rho, float sigma){
	T *d_Pinv, * d_Sn, * d_Sbd;
    gpuErrchk(cudaMalloc(&d_Pinv, 3*STATE_SIZE*STATE_SIZE*KNOT_POINTS*sizeof(T)));
	gpuErrchk(cudaMalloc(&d_Sn, 3*STATE_SIZE*STATE_SIZE*KNOT_POINTS*sizeof(T)));
	gpuErrchk(cudaMalloc(&d_Sbd, 3*STATE_SIZE*STATE_SIZE*KNOT_POINTS*sizeof(T)));


	/* form_schur */
	form_schur(d_Sn, prob->d_H, prob->d_A, rho, sigma);

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

	/*update z*/
	update_z<T>(prob->d_A, d_x, d_lambda, d_z, prob->d_l, prob->d_u, rho);

	/*update lambda*/
	update_lambda<T>(prob->d_A, d_x, d_lambda, d_z, rho);
}