#include <cublas_v2.h>
#include <cooperative_groups.h>
#include <admm_iter.cuh>



template <typename T>
void admm_solve(qp<T> *prob, T * d_x,  T *d_lambda, T *d_z, float rho, float sigma =1e-6, float tol =1e-3, int max_iter=1000, int update_rho=1){

	float primal_res_value, dual_res_value;
	
	for(int iter=0;  iter<max_iter; iter++){
		admm_iter(prob, d_x, d_lambda, d_z, rho, sigma);

		primal_res_value = primal_res(prob->d_A, d_x, d_z);
		dual_res_value = dual_res(prob->d_A, prob->d_H, prob->d_g, d_x, d_lambda);

		if (primal_res_value < tol && dual_res_value < tol){
			std::cout<< "Finished in iters: " << iter + 1 << " with primal res:" << primal_res_value << " dual res:" << dual_res_value <<"\n\n\n";
			break;
		}

		if (update_rho) {
			if (primal_res_value > 10 * dual_res_value){
			rho = 2 * rho;
			}
			else if (dual_res_value > 10 * primal_res_value) {
				rho = 1/2 * rho;
			}
			if ( rho < 1e-6) rho = 1e-6;
			else if ( rho > 1e6 ) rho = 1e6;
		}

		// T h_x[NX];
		// T h_z[NC];
		// T h_lambda[NC];
		// gpuErrchk(cudaMemcpy(h_x, d_x, NX * sizeof(T), cudaMemcpyDeviceToHost));
		// gpuErrchk(cudaMemcpy(h_lambda, d_lambda, NC * sizeof(T), cudaMemcpyDeviceToHost));
		// gpuErrchk(cudaMemcpy(h_z, d_z, NC * sizeof(T), cudaMemcpyDeviceToHost));

		// std::cout<< "ADMM ITER iter: " << iter<< " with primal res:" << primal_res_value << " dual res:" << dual_res_value <<"\n\n\n";
			
		// std::cout << "X: ";
		// for(int i=0; i<NX; i++){
		// 	std::cout << h_x[i] << " ";
		// }
		// std::cout << "\n\n";
		// std::cout << "lambda: ";
		// for(int i=0; i<NC; i++){
		// 	std::cout << h_lambda[i] << " ";
		// }

		// std::cout << "\n\n";
		// std::cout << "z: ";
		// for(int i=0; i<NC; i++){
		// 	std::cout << h_z[i] << " ";
		// }

		// std::cout << "\n\n";


	}

	
}


template <typename T>
void admm_solve_outer(T * h_H,  T *h_g, T *h_A, T * h_l , T * h_u,  T * h_x,  T *h_lambda, T *h_z, float rho, float sigma =1e-6, float tol =1e-3, int max_iters=1000, int update_rho=1){

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
	admm_solve(&prob, d_x, d_lambda, d_z, rho, sigma, tol, max_iters, update_rho);

	/* Copy x, lambda, z */
	gpuErrchk(cudaMemcpy(h_x, d_x, NX * sizeof(T), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_lambda, d_lambda, NC * sizeof(T), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_z, d_z, NC * sizeof(T), cudaMemcpyDeviceToHost));

	gpuErrchk(cudaFree(d_H));
	gpuErrchk(cudaFree(d_A));
	gpuErrchk(cudaFree(d_g));
	gpuErrchk(cudaFree(d_l));
	gpuErrchk(cudaFree(d_u));
	gpuErrchk(cudaFree(d_x));
	gpuErrchk(cudaFree(d_lambda));
	gpuErrchk(cudaFree(d_z));
	
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
