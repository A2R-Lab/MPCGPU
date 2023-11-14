#include <cublas_v2.h>
#include <cooperative_groups.h>
#include <admm_iter.cuh>



template <typename T>
void admm_solve(qp<T> *prob, T * d_x,  T *d_lambda, T *d_z, float rho, float sigma =1e-6, float tol =1e-3, int max_iter=1000, int update_rho=1){

	float primal_res_value, dual_res_value;
	float primal_res_ptr[1], dual_res_ptr[1];
	
	for(int iter=0;  iter<max_iter; iter++){
		admm_iter(prob, d_x, d_lambda, d_z, rho, sigma);


		res(prob->d_A, prob->d_H, prob->d_g, d_x, d_lambda, d_z, primal_res_ptr, dual_res_ptr);

		primal_res_value = abs(primal_res_ptr[0]);
		dual_res_value = abs(dual_res_ptr[0]);

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

#if DEBUG_MODE
		T h_x[NX];
		T h_z[NC];
		T h_lambda[NC];
		gpuErrchk(cudaMemcpy(h_x, d_x, NX * sizeof(T), cudaMemcpyDeviceToHost));
		gpuErrchk(cudaMemcpy(h_lambda, d_lambda, NC * sizeof(T), cudaMemcpyDeviceToHost));
		gpuErrchk(cudaMemcpy(h_z, d_z, NC * sizeof(T), cudaMemcpyDeviceToHost));

		std::cout<< "ADMM ITER iter: " << iter<< " with primal res:" << primal_res_value << " dual res:" << dual_res_value <<"\n\n\n";
			
		std::cout << "X: ";
		for(int i=0; i<NX; i++){
			std::cout << h_x[i] << " ";
		}
		std::cout << "\n\n";
		std::cout << "lambda: ";
		for(int i=0; i<NC; i++){
			std::cout << h_lambda[i] << " ";
		}

		std::cout << "\n\n";
		std::cout << "z: ";
		for(int i=0; i<NC; i++){
			std::cout << h_z[i] << " ";
		}

		std::cout << "\n\n";
#endif 


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