#include <cublas_v2.h>
#include <cooperative_groups.h>

cublasHandle_t handle;
cublasCreate(&handle);


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
form_schur(T * d_S, T * d_H, T *d_A,  float rho, float sigma){
	/* S = H + sigma * I + rho * A.T * A */


	/* Allocating memory*/
	T * d_I, d_Anorm;
	gpuErrchk(cudaMalloc(d_I, NX * NX * sizeof(T)));
	gpuErrchk(cudaMalloc(d_Aorm, NX * NX * sizeof(T)));


	/* TODO: create sigma Identity matrix*/
	createIdentityMatrix<<<NX, NX>>>(d_I, NX);

	/* Anorm = A.T * A */
	cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, NX, NX, NC, 1, d_A, NX, 1, d_A, NC, d_Anorm, NX);

	/* S = H + sigma * I */
	cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, NX, NX, 1.0, d_H, NX, sigma, d_I, NX, d_S, NX);

	/* S = S + rho * Anorm */
	cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, NX, NX, 1.0, d_S, NX, rho, d_Anorm, NX, d_S, NX);

}

template <typename T>
convert_to_bd(T * d_Sn, t * d_Sbd){
	/* TODO: launch*/
	convert_to_bd_kernel<<<KNOT_POINTS, 3 * STATE_SIZE * STATE_SIZE>>>(d_Sn, d_Sbd);

}

/* 
for (int i = 0; i <  3 * state_size ; i++){
		for (int j = 0; j < state_size ; j++){

			

				
				if (blockIdx.x == 0 && i * state_size + j < state_size * state_size){
					d_Sbd[offset_bd + i * state_size + j] = 0;
				}

				else if (blockIdx.x == blockDim.x - 1 && i * state_size + j >= 2 * state_size * state_size) {
					d_Sbd[offset_bd + i * state_size + j] = 0;
				}
				
				d_Sbd[offset_bd + i * state_size + j] = d_Sn[offset_n + j]
			
		}
		offset_n += state_size * knot_points;
	}
*/
/* TODO: only for blocks==knot_points */
template <typename T>
__global__
convert_to_bd_kernel(T * d_Sn, t * d_Sbd){
	int j = threadIdx.x % state_size;
	int i = (threadIdx.x - j) / state_size;

	offset_bd = 3 * state_size * state_size * blockIdx.x;
	offset_n = blockIdx.x * ( state_size )  + ( blockIdx.x - 1 ) * state_size * state_size * knot_points;
	offset_n += i * state_size * knot_points;
	
	if ( blockIdx.x == 0 && threadIdx.x < state_size * state_size){
		d_Sbd[offset_bd + threadIdx.x] = 0;
	}
	
	else if ( blockIdx.x == blockDim.x - 1 && threadIdx.x >= 2 * state_size * state_size){
		d_Sbd[offset_bd + threadIdx.x] = 0;
	}

	else {
		d_Sbd[offset_bd + threadIdx.x] = d_Sn[offset_n + j];
	}

}

form_ss(T * d_Pinv, T * d_S){
	/* Launch global function */
    void *ss_kernel = (void *) form_ss_kernel<T, STATE_SIZE, KNOT_POINTS>;

    void *kernelArgs[] = {
        (void *)&d_S,
        (void *)&d_Pinv,
    };

    size_t ss_kernel_smem_size = ssSharedMemSize<T>(state_size, knot_points);


    gpuErrchk(cudaLaunchCooperativeKernel(ss_kernel, knot_points, 64, kernelArgs, ss_kernel_smem_size));    
    gpuErrchk(cudaPeekAtLastError());
}


template <typename T>
__global__
form_ss_kernel(T * d_Pinv, T * d_S){
	extern __shared__ T s_temp[];
	int state_size = STATE_SIZE;
	int knot_points = KNOT_POINTS;
    
	for(unsigned ind=blockIdx.x; ind<knot_points; ind+=gridDim.x){
		gato_ss_from_schur<T>(
			state_size, knot_points,
			d_S,
			d_Pinv,
			s_temp,
			ind
		);
	}
	grid.sync();

    for(unsigned ind=blockIdx.x; ind<knot_points; ind+=gridDim.x){
        gato_form_ss_inner<T>(
            state_size, knot_points,
            d_S,
            d_Pinv,
            d_gamma,
            s_temp,
            ind
        );
    }
    grid.sync();
}

template <typename T>
admm_solve(qp *prob, T * d_x,  T *d_lambda, T *d_z, float rho, float sigma =1e-6, float tol =1e-3, max_iter=1000){

	/* Allocate memory for schur and pinv */
	int state_size = STATE_SIZE;
	int knot_points = KNOT_POINTS;

	T *d_Pinv, d_S, d_Sbd;
    gpuErrchk(cudaMalloc(&d_Pinv, 3*states_size*state_size*knot_points*sizeof(T)));
	gpuErrchk(cudaMalloc(&d_S, 3*states_size*state_size*knot_points*sizeof(T)));
	gpuErrchk(cudaMalloc(&d_Sbd, 3*states_size*state_size*knot_points*sizeof(T)));

	/* form_schur */
	form_schur(d_S, prob->d_H,prob->d_A, sigma, rho);

	/* convert to custom bd form */
	convert_to_bd(d_S, d_Sbd);

	/* TODO: form_precon from schur */
	form_ss(d_Pinv, d_Sbd);

	for(int iter=0; iter++; iter<max_iter){
		admm_iter(prob, d_Sbd, d_Pinv, d_x, d_lambda, rho);
	}
}


template <typename T>
admm_solve_outer(T * h_H,  T *h_g, T *h_A, T*h_l , T*d_u, int nx, int nc, T * h_x, T *h_z, T *h_lambda, float rho, float sigma =1e-6, float tol =1e-3, max_iter=1000){

	/*Allocate memory for device pointers */
	T * d_H, d_g, d_A, d_l, d_u, d_x, d_lambda, d_z;

	int msize = MSIZE;
	gpuErrchk(cudaMalloc(d_H, NX * NX * sizeof(T)));
	gpuErrchk(cudaMalloc(d_A, NC * NX * sizeof(T)));


	gpuErrchk(cudaMalloc(d_g, NX * sizeof(T)));
	gpuErrchk(cudaMalloc(d_l, NC * sizeof(T)));
	gpuErrchk(cudaMalloc(d_u, NC * sizeof(T)));
	gpuErrchk(cudaMalloc(d_x, NX * sizeof(T)));
	gpuErrchk(cudaMalloc(d_lambda, NC * sizeof(T)));
	gpuErrchk(cudaMalloc(d_z, NC * sizeof(T)));


	/* Copy from host to device memory */
	gpuErrchk(cudaMemcpy(d_H, h_H,  NX * NX * sizeof(T), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_A, h_A, NC * NX * sizeof(T), cudaMemcpyHostToDevice));

	gpuErrchk(cudaMemcpy(d_g, h_g, NX * sizeof(T), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_l, h_l, NC * sizeof(T), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_u, h_u, NC * sizeof(T), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_x, h_x, NX * sizeof(T), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_lamda, h_lambda, NC * sizeof(T), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_z, h_z, NC * sizeof(T), cudaMemcpyHostToDevice));


	/* Make QP struct */
	qp<T> prob(d_H, d_g, d_A, d_l, d_u, NX, NC);


	/* Call admm_solve */
	admm_solve(&prob, d_x, d_lambda, d_z, rho, sigma, tol, max_iters);

	/* Copy x, lambda, z */
	gpuErrchk(cudaMemcpy(h_x, d_x, NX * sizeof(T), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_lamda, d_lambda, NC * sizeof(T), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_z, d_z, NC * sizeof(T), cudaMemcpyDeviceToHost));
}
