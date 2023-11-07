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


	/*  create sigma Identity matrix*/
	createIdentityMatrix<<<NX, NX>>>(d_I, NX);

	/* Anorm = A.T * A */
	float one = 1.0f;
	float beta = 0.0f;

	/* TODO: understand leading dimension*/
	cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, NX, NX, NC, &one, d_A, NC, d_A, NC, &beta, d_Anorm, NX);

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

	int state_size = STATE_SIZE;
	int knot_points = KNOT_POINTS;
	int j = threadIdx.x % state_size;
	int i = (threadIdx.x - j) / state_size;

	int offset_bd = 3 * state_size * state_size * blockIdx.x;
	int offset_n = blockIdx.x * ( state_size )  + ( blockIdx.x - 1 ) * state_size * state_size * knot_points;
	offset_n += i * state_size * knot_points + j;
	
	if ( blockIdx.x == 0 && threadIdx.x < state_size * state_size){
		d_Sbd[offset_bd + threadIdx.x] = 0;
	}
	
	else if ( blockIdx.x == blockDim.x - 1 && threadIdx.x >= 2 * state_size * state_size){
		d_Sbd[offset_bd + threadIdx.x] = 0;
	}

	else {
		d_Sbd[offset_bd + threadIdx.x] = d_Sn[offset_n];
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
        (void *)&d_Pinv,
        (void *)&d_S,
    };

    size_t pcg_kernel_smem_size = pcgSharedMemSize<T>(STATE_SIZE, KNOT_POINTS);


    gpuErrchk(cudaLaunchCooperativeKernel(ss_kernel, KNOT_POINTS, 64, kernelArgs, pcg_kernel_smem_size));    
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

template <typename T>
void admm_solve(qp<T> *prob, T * d_x,  T *d_lambda, T *d_z, float rho, float sigma =1e-6, float tol =1e-3, int max_iter=1000){

	/* Allocate memory for schur and pinv */

	float primal_res_value, dual_res_value;
	
	max_iter = 100;
	for(int iter=0;  iter<max_iter; iter++){
		admm_iter(prob, d_x, d_lambda, d_z, rho, sigma);

		primal_res_value = primal_res(prob->d_A, d_x, d_z);
		dual_res_value = dual_res(prob->d_A, prob->d_H, prob->d_g, d_x, d_lambda);

		if (primal_res_value < tol && dual_res_value < tol){
			std::cout<< "Finished in iters: " << iter + 1 << " with primal res:" << primal_res_value << " dual res:" << dual_res_value <<"\n\n\n";
			break;
		}

		if (primal_res_value > 10 * dual_res_value){
			rho = 2 * rho;
		}
		else if (dual_res_value > 10 * primal_res_value) {
			rho = 1/2 * rho;
		}
		if ( rho < 1e-6) rho = 1e-6;
		else if ( rho > 1e6 ) rho = 1e6;


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


	/* 
	T h_Anorm[9];
	T h_A[15];
	// gpuErrchk(cudaMalloc(&h_Anorm, NX * NX * sizeof(T)));
	gpuErrchk(cudaMemcpy(h_Anorm, d_Anorm,  NX * NX * sizeof(T), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_A, d_A,  NC * NX * sizeof(T), cudaMemcpyDeviceToHost));
	std::cout << "Anorm: ";
	for(int i=0; i<NX*NX; i++){
		std::cout << h_Anorm[i] << " ";
	}
	std::cout << "\n\n";

	std::cout << "A: ";
	for(int i=0; i<NC*NX; i++){
		std::cout << h_A[i] << " ";
	}
	std::cout << "\n\n";

	*/

		// T h_gamma[NX];
	// gpuErrchk(cudaMemcpy(h_gamma, d_gamma, NX * sizeof(T), cudaMemcpyDeviceToHost));
	// std::cout << "Gamma: ";
	// 	for(int i=0; i<9; i++){
	// 		std::cout << h_gamma[i] << " ";
	// 	}


	
	// if (threadIdx.x == 0 && blockIdx.x == 0){
	// 	printf("d_Sn: ");
	// 	for(int k=0;  k<NX ; k++){
	// 		for(int l=0; l<NX; l++)
	// 			printf("%f ", d_Sn[k + l*NX]);
	// 		printf("\n");
	// 	}
	// 	printf("\n\n");
		
	// }


		// 	T h_x[NX];
		// T h_z[NC];
		// T h_lambda[NC];
		// gpuErrchk(cudaMemcpy(h_x, d_x, NX * sizeof(T), cudaMemcpyDeviceToHost));
		// gpuErrchk(cudaMemcpy(h_lambda, d_lambda, NC * sizeof(T), cudaMemcpyDeviceToHost));
		// gpuErrchk(cudaMemcpy(h_z, d_z, NC * sizeof(T), cudaMemcpyDeviceToHost));
		// std::cout<< "ADMM ITER: " << iter <<"\n\n\n";
		// std::cout << "X: ";
		// for(int i=0; i<9; i++){
		// 	std::cout << h_x[i] << " ";
		// }
		// std::cout << "\n\n";
		// std::cout << "lambda: ";
		// for(int i=0; i<9; i++){
		// 	std::cout << h_lambda[i] << " ";
		// }

		// std::cout << "\n\n";
		// std::cout << "z: ";
		// for(int i=0; i<9; i++){
		// 	std::cout << h_z[i] << " ";
		// }

		// std::cout << "\n\n";


	// if (threadIdx.x == 0 && blockIdx.x == 0){
	// 	printf("d_Pinv:\n");

	// 	for(int k=0; k<KNOT_POINTS;k++){
	// 		int offset = k * 3 * STATE_SIZE * STATE_SIZE;
	// 		for (int i=0; i < STATE_SIZE; i++){
	// 			for (int j=0; j < 3 * STATE_SIZE; j++){
	// 				printf("%f ", d_Pinv[offset + i + j * STATE_SIZE]);
	// 			}
	// 			printf("\n");
	// 		}
	// 	}
	// 	printf("\n\n");
		
	// }

		// if (threadIdx.x == 0 && blockIdx.x == 0){
	// 	printf("d_Sbd:\n");

	// 	for(int k=0; k<KNOT_POINTS;k++){
	// 		int offset = k * 3 * STATE_SIZE * STATE_SIZE;
	// 		for (int i=0; i < STATE_SIZE; i++){
	// 			for (int j=0; j < 3 * STATE_SIZE; j++){
	// 				printf("%f ", d_S[offset + i + j * STATE_SIZE]);
	// 			}
	// 			printf("\n");
	// 		}
	// 	}
	// 	printf("\n\n");
		
	// }


		// if(threadIdx.x==0 && blockIdx.x == 0){
	// 	printf("\nres dot");
	// 	for(int i=0;i<NC;i++){
	// 		printf("%f ", d_Axz[i]);
	// 	};
	// 	printf("\n");
	// }

		// grid.sync();
	// if(threadIdx.x==0 && blockIdx.x == 0){
	// 	printf("\nx: ");
	// 	for(int i=0;i<NX;i++){
	// 		printf("%f ", d_x[i]);
	// 	};
	// 	printf("\n");
	// }
	// if(threadIdx.x==0 && blockIdx.x == 0){
	// 	printf("\nHx: ");
	// 	for(int i=0;i<NX;i++){
	// 		printf("%f ", d_Hx[i]);
	// 	};
	// 	printf("\n");
	// }

	// grid.sync();
	// if(threadIdx.x==0 && blockIdx.x == 0){
	// 	printf("\nAtl: ");
	// 	for(int i=0;i<NX;i++){
	// 		printf("%f ", d_Atl[i]);
	// 	};
	// 	printf("\n");
	// };

		// grid.sync();
	// if(threadIdx.x==0 && blockIdx.x == 0){
	// 	printf("\nr: ");
	// 	for(int i=0;i<NX;i++){
	// 		printf("%f ", d_res[i]);
	// 	};
	// 	printf("\n");
	// };
