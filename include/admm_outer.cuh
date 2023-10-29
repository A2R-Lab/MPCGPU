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
	gpuErrchk(cudaMalloc(d_I, nx * nx * sizeof(T)));
	gpuErrchk(cudaMalloc(d_Aorm, nx * nx * sizeof(T)));


	/* TODO: create sigma Identity matrix*/
	createIdentityMatrix<<<nx, nx>>>(d_I, nx);

	/* Anorm = A.T * A */
	cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, nx, nx, &alpha, d_A, nx, &beta, d_A, nx, d_Anorm, nx);

	/* S = H + sigma * I */
	cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, 1.0, d_H, n, sigma, d_I, n, d_S, n);

	/* S = S + rho * Anorm */
	cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, 1.0, d_S, n, rho, d_Anorm, n, d_S, n);

}

convert_to_bd(T * d_S){
	/* TODO: */
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

	/* TODO: Allocate memory for schur and pinv */

	/* form_schur */
	form_schur(d_S, prob->d_H,prob->d_A, sigma, rho);

	/* convert to custom bd form */
	convert_to_bd(d_Sbd);

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

	int msize = (state_size + control_size) * knot_points;
	gpuErrchk(cudaMalloc(d_H, msize * msize * sizeof(T)));
	gpuErrchk(cudaMalloc(d_A, msize * msize * sizeof(T)));


	gpuErrchk(cudaMalloc(d_g, msize * sizeof(T)));
	gpuErrchk(cudaMalloc(d_l, msize * sizeof(T)));
	gpuErrchk(cudaMalloc(d_u, msize * sizeof(T)));
	gpuErrchk(cudaMalloc(d_x, msize * sizeof(T)));
	gpuErrchk(cudaMalloc(d_lambda, msize * sizeof(T)));
	gpuErrchk(cudaMalloc(d_z, msize * sizeof(T)));


	/* Copy from host to device memory */
	gpuErrchk(cudaMemcpy(d_H, h_H, msize * msize * sizeof(T), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_A, h_A, msize * msize * sizeof(T), cudaMemcpyHostToDevice));

	gpuErrchk(cudaMemcpy(d_g, h_g, msize * sizeof(T), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_l, h_l, msize * sizeof(T), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_u, h_u, msize * sizeof(T), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_x, h_x, msize * sizeof(T), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_lamda, h_lambda, msize * sizeof(T), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_z, h_z, msize * sizeof(T), cudaMemcpyHostToDevice));


	/* Make QP struct */
	qp<T> prob(d_H, d_g, d_A, d_l, d_u, nx, nc);


	/* Call admm_solve */
	admm_solve(&prob, d_x, d_lambda, d_z, rho, sigma, tol, max_iters);

	/* Copy x, lambda, z */
	gpuErrchk(cudaMemcpy(h_x, d_x, msize * sizeof(T), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_lamda, d_lambda, msize * sizeof(T), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_z, d_z, msize * sizeof(T), cudaMemcpyDeviceToHost));
}
