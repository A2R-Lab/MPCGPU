#include <cublas_v2.h>
#include <cooperative_groups.h>
#include <admm_iter.cuh>

template <typename T>
struct qp{
	T *d_H, *d_g, *d_A, *d_l , *d_u;
	int nx, nc;

	qp(T *d_H, 
		T *d_g, 
		T *d_A, 
		T *d_l, 
		T *d_u, 
		int nx, 
		int nc):
		d_H(d_H), d_g(d_g), d_A(d_A), d_l(d_l), d_u(d_u), nx(nx), nc(nc) {}
};

template <typename T>
void admm_iter(qp<T> *prob, T *d_S, T *d_Pinv, T *d_x, T *d_lambda, T *d_z, float rho, float sigma){

	/*Allocate memory for gamma, */
	T *d_gamma;
	gpuErrchk(cudaMalloc(&d_gamma, NX * sizeof(T)));

	/*compute gamma*/
	compute_gamma<T>(d_gamma, prob->d_g, prob->d_A, d_x, d_lambda, d_z, rho, sigma);

	/*call pcg*/
	solve_pcg<T>(d_S, d_Pinv, d_gamma, d_x);

	/*update z*/
	update_z<T>(prob->d_A, d_x, d_lambda, d_z, prob->d_l, prob->d_u, rho);

	/*update lambda*/
	update_lambda<T>(prob->d_A, d_x, d_lambda, d_z, rho);
}

template <typename T>
__global__ void createIdentityMatrix(T* A, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * n) {
        int row = idx / n;
        int col = idx % n;
        A[idx] = (row == col) ? 1.0f : 0.0f;
    }
}

template <typename T>
void form_schur(T * d_S, T * d_H, T *d_A,  float rho, float sigma){
	cublasHandle_t handle;
	cublasCreate(&handle);

	/* S = H + sigma * I + rho * A.T * A */


	/* Allocating memory*/
	T * d_I, * d_Anorm;
	gpuErrchk(cudaMalloc(&d_I, NX * NX * sizeof(T)));
	gpuErrchk(cudaMalloc(&d_Anorm, NX * NX * sizeof(T)));


	/* TODO: create sigma Identity matrix*/
	createIdentityMatrix<<<NX, NX>>>(d_I, NX);

	/* Anorm = A.T * A */
	float one = 1.0f;
	cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, NX, NX, NC, &one, d_A, NX, d_A, NC, 0, d_Anorm, NX);

	/* S = H + sigma * I */
	cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, NX, NX, &one, d_H, NX, &sigma, d_I, NX, d_S, NX);

	/* S = S + rho * Anorm */
	cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, NX, NX, &one, d_S, NX, &rho, d_Anorm, NX, d_S, NX);

}


/* 
for (int i = 0; i <  3 * STATE_SIZE ; i++){
		for (int j = 0; j < STATE_SIZE ; j++){

			

				
				if (blockIdx.x == 0 && i * STATE_SIZE + j < STATE_SIZE * STATE_SIZE){
					d_Sbd[offset_bd + i * STATE_SIZE + j] = 0;
				}

				else if (blockIdx.x == blockDim.x - 1 && i * STATE_SIZE + j >= 2 * STATE_SIZE * STATE_SIZE) {
					d_Sbd[offset_bd + i * STATE_SIZE + j] = 0;
				}
				
				d_Sbd[offset_bd + i * STATE_SIZE + j] = d_Sn[offset_n + j]
			
		}
		offset_n += STATE_SIZE * KNOT_POINTS;
	}
*/
/* TODO: only for blocks==KNOT_POINTS */
template <typename T>
__global__
void convert_to_bd_kernel(T * d_Sn, T * d_Sbd){
	int j = threadIdx.x % STATE_SIZE;
	int i = (threadIdx.x - j) / STATE_SIZE;

	int offset_bd = 3 * STATE_SIZE * STATE_SIZE * blockIdx.x;
	int offset_n = blockIdx.x * ( STATE_SIZE )  + ( blockIdx.x - 1 ) * STATE_SIZE * STATE_SIZE * KNOT_POINTS;
	offset_n += i * STATE_SIZE * KNOT_POINTS;
	
	if ( blockIdx.x == 0 && threadIdx.x < STATE_SIZE * STATE_SIZE){
		d_Sbd[offset_bd + threadIdx.x] = 0;
	}
	
	else if ( blockIdx.x == blockDim.x - 1 && threadIdx.x >= 2 * STATE_SIZE * STATE_SIZE){
		d_Sbd[offset_bd + threadIdx.x] = 0;
	}

	else {
		d_Sbd[offset_bd + threadIdx.x] = d_Sn[offset_n + j];
	}

}


template <typename T>
void convert_to_bd(T * d_Sn, T * d_Sbd){
	/* TODO: launch*/
	convert_to_bd_kernel<<<KNOT_POINTS, 3 * STATE_SIZE * STATE_SIZE>>>(d_Sn, d_Sbd);

}

template <typename T>
__global__
void form_ss_kernel(T * d_Pinv, T * d_S){
	extern __shared__ T s_temp[];
    const cgrps::grid_group grid = cgrps::this_grid();
    
	for(unsigned ind=blockIdx.x; ind<KNOT_POINTS; ind+=gridDim.x){
		gato_ss_from_schur<T>(
			STATE_SIZE, KNOT_POINTS,
			d_S,
			d_Pinv,
			s_temp,
			ind
		);
	}
	grid.sync();

    for(unsigned ind=blockIdx.x; ind<KNOT_POINTS; ind+=gridDim.x){
        gato_form_ss_inner<T>(
            STATE_SIZE, KNOT_POINTS,
            d_S,
            d_Pinv,
            s_temp,
            ind
        );
    }
    grid.sync();
}

template <typename T>
void form_ss(T * d_Pinv, T * d_S){
	/* Launch global function */
    void *ss_kernel = (void *) form_ss_kernel<T>;

    void *kernelArgs[] = {
        (void *)&d_S,
        (void *)&d_Pinv,
    };

    size_t pcg_kernel_smem_size = pcgSharedMemSize<T>(STATE_SIZE, KNOT_POINTS);


    gpuErrchk(cudaLaunchCooperativeKernel(ss_kernel, KNOT_POINTS, 64, kernelArgs, pcg_kernel_smem_size));    
    gpuErrchk(cudaPeekAtLastError());
}



template <typename T>
void admm_solve(qp<T> *prob, T * d_x,  T *d_lambda, T *d_z, float rho, float sigma =1e-6, float tol =1e-3, int max_iter=1000){

	/* Allocate memory for schur and pinv */

	T *d_Pinv, * d_S, * d_Sbd;
    gpuErrchk(cudaMalloc(&d_Pinv, 3*STATE_SIZE*STATE_SIZE*KNOT_POINTS*sizeof(T)));
	gpuErrchk(cudaMalloc(&d_S, 3*STATE_SIZE*STATE_SIZE*KNOT_POINTS*sizeof(T)));
	gpuErrchk(cudaMalloc(&d_Sbd, 3*STATE_SIZE*STATE_SIZE*KNOT_POINTS*sizeof(T)));

	/* form_schur */
	form_schur(d_S, prob->d_H, prob->d_A, sigma, rho);

	/* convert to custom bd form */
	convert_to_bd(d_S, d_Sbd);

	/* TODO: form_precon from schur */
	form_ss(d_Pinv, d_Sbd);

	for(int iter=0;  iter<max_iter; iter++){
		admm_iter(prob, d_Sbd, d_Pinv, d_x, d_lambda, d_z, rho, sigma);
	}
}


template <typename T>
void admm_solve_outer(T * h_H,  T *h_g, T *h_A, T * h_l , T * h_u,  T * h_x,  T *h_lambda, T *h_z, float rho, float sigma =1e-6, float tol =1e-3, int max_iters=1000){

	/*Allocate memory for device pointers */
	T * d_H, * d_g, * d_A, * d_l, * d_u, * d_x, * d_lambda, * d_z;

	gpuErrchk(cudaMalloc(&d_H, NX * NX * sizeof(T)));
	gpuErrchk(cudaMalloc(&d_A, NC * NX * sizeof(T)));


	gpuErrchk(cudaMalloc(&d_g, NX * sizeof(T)));
	gpuErrchk(cudaMalloc(&d_l, NC * sizeof(T)));
	gpuErrchk(cudaMalloc(&d_u, NC * sizeof(T)));
	gpuErrchk(cudaMalloc(&d_x, NX * sizeof(T)));
	gpuErrchk(cudaMalloc(&d_lambda, NC * sizeof(T)));
	gpuErrchk(cudaMalloc(&d_z, NC * sizeof(T)));


	/* Copy from host to device memory */
	gpuErrchk(cudaMemcpy(d_H, h_H,  NX * NX * sizeof(T), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_A, h_A, NC * NX * sizeof(T), cudaMemcpyHostToDevice));

	gpuErrchk(cudaMemcpy(d_g, h_g, NX * sizeof(T), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_l, h_l, NC * sizeof(T), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_u, h_u, NC * sizeof(T), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_x, h_x, NX * sizeof(T), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_lambda, h_lambda, NC * sizeof(T), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_z, h_z, NC * sizeof(T), cudaMemcpyHostToDevice));


	/* Make QP struct */
	struct qp<T> prob(d_H, d_g, d_A, d_l, d_u, NX, NC);


	/* Call admm_solve */
	admm_solve(&prob, d_x, d_lambda, d_z, rho, sigma, tol, max_iters);

	/* Copy x, lambda, z */
	gpuErrchk(cudaMemcpy(h_x, d_x, NX * sizeof(T), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_lambda, d_lambda, NC * sizeof(T), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_z, d_z, NC * sizeof(T), cudaMemcpyDeviceToHost));
}
