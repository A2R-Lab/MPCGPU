#include <glass.cuh>
#include <gpuassert.cuh>
#include <interface.cuh>
#include <types.cuh>
#include <cmath>

#define NUM_THREADS 64
#define KNOT_POINTS 3
#define NX 9
#define NC 9
#define STATE_SIZE NX/KNOT_POINTS

template <typename T>
T compute_norm(T * h_x, int n){
	T max_value = h_x[0];
	for (int i=0; i<n; i++){
		if(abs(h_x[i]) > max_value){
			max_value = abs(h_x[i]);
		}
	}
	return max_value;
}

template <typename T>
__global__
 void primal_res_kernel(T * d_A, T * d_x, T *d_z, T * d_Ax, T * d_Axz){
	/* ||Ax - z||*/

	/* Ax = A * x */
	glass::gemv<T, false>(NC, NX, 1, d_A, d_x, d_Ax);

	/* Axz = Ax - z */
	glass::axpby<T>(NC, 1, d_Ax, -1, d_z, d_Axz);
}

template <typename T>
 float primal_res(T * d_A, T * d_x, T *d_z){
	/* Alloc d_Ax, d_Axz*/
	T *d_Ax, *d_Axz;
	gpuErrchk(cudaMalloc(&d_Ax, NC * sizeof(T)));
	gpuErrchk(cudaMalloc(&d_Axz, NC * sizeof(T)));

	/* Launch */
	void *primal_res_kernel_ptr = (void *) primal_res_kernel<T>;

	void *kernelArgs[] = {
		(void *)&d_A, 
		(void *)&d_x, 
		(void *)&d_z, 
		(void *)&d_Ax, 
		(void *)&d_Axz
	};

	gpuErrchk(cudaLaunchCooperativeKernel(primal_res_kernel_ptr, KNOT_POINTS, NUM_THREADS, kernelArgs, 0));    
    gpuErrchk(cudaPeekAtLastError());
	/* Launch*/

	/* prima_res = norm(primal_res) */
	T primal_res_value[NC];
	gpuErrchk(cudaMemcpy(primal_res_value, d_Axz, NC * sizeof(T), cudaMemcpyDeviceToHost));
	return compute_norm<T>(primal_res_value, NC);

 }


template <typename T>
__global__
 void dual_res_kernel(T * d_A, T * d_H, T * d_g, T *d_x,  T * d_lambda, T * d_Hx, T * d_Atl, T * d_res){
	/* || H*x + g + A.T*lamb ||*/

	const cgrps::grid_group grid = cgrps::this_grid();

	/* Hx = H * x */
	glass::gemv<T, false>(NX, NX, 1, d_H, d_x, d_Hx);

	/* Atl = A.T * lamb */
	glass::gemv<T, true>(NC, NX, 1, d_A, d_lambda, d_Atl);

	/* res = Hx + Atl */
	glass::axpby<T>(NX, 1, d_Hx, 1, d_Atl, d_res);

	/* res += g */
	glass::axpy<T>(NX, 1, d_g, d_res);

	

}

template <typename T>
 float dual_res(T * d_A, T * d_H, T * d_g,  T * d_x, T * d_lambda ){
	/* Alloc:T * d_Hx, T * d_Atl, T * d_res*/
	T *d_Hx, *d_Atl, *d_res;
	gpuErrchk(cudaMalloc(&d_Hx, NX * sizeof(T)));
	gpuErrchk(cudaMalloc(&d_Atl, NX * sizeof(T)));
	gpuErrchk(cudaMalloc(&d_res, NX * sizeof(T)));

	/* Launch */
	void *dual_res_kernel_ptr = (void *) dual_res_kernel<T>;

	void *kernelArgs[] = {
		(void *)&d_A, 
		(void *)&d_H, 
		(void *)&d_g,
		(void *)&d_x, 
		(void *)&d_lambda, 
		(void *)&d_Hx, 
		(void *)&d_Atl, 
		(void *)&d_res
	};

	gpuErrchk(cudaLaunchCooperativeKernel(dual_res_kernel_ptr, KNOT_POINTS, NUM_THREADS, kernelArgs, 0));    
    gpuErrchk(cudaPeekAtLastError());

	/* dual_res = norm(dual_res) */
	T dual_res_value[NX];
	gpuErrchk(cudaMemcpy(dual_res_value, d_res, NX * sizeof(T), cudaMemcpyDeviceToHost));
	return compute_norm<T>(dual_res_value, NX);
}


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
void solve_pcg(T * d_S, T * d_Pinv, T *d_gamma,  T * d_x){
	pcg_config config;
	config.empty_pinv = 0;
    
    /*   PCG vars   */
    T  *d_r, *d_p, *d_v_temp, *d_eta_new_temp;// *d_r_tilde, *d_upsilon;
    gpuErrchk(cudaMalloc(&d_r, STATE_SIZE*KNOT_POINTS*sizeof(T)));
    gpuErrchk(cudaMalloc(&d_p, STATE_SIZE*KNOT_POINTS*sizeof(T)));
    gpuErrchk(cudaMalloc(&d_v_temp, KNOT_POINTS*sizeof(T)));
    gpuErrchk(cudaMalloc(&d_eta_new_temp, KNOT_POINTS*sizeof(T)));
    
    
    
    void *pcg_kernel = (void *) pcg<T, STATE_SIZE, KNOT_POINTS>;
    uint32_t *d_pcg_iters;
    gpuErrchk(cudaMalloc(&d_pcg_iters, sizeof(uint32_t)));
    bool *d_pcg_exit;
    gpuErrchk(cudaMalloc(&d_pcg_exit, sizeof(bool)));
    
    void *pcgKernelArgs[] = {
        (void *)&d_S,
        (void *)&d_Pinv,
        (void *)&d_gamma, 
        (void *)&d_x,
        (void *)&d_r,
        (void *)&d_p,
        (void *)&d_v_temp,
        (void *)&d_eta_new_temp,
        (void *)&d_pcg_iters,
        (void *)&d_pcg_exit,
        (void *)&config.pcg_max_iter,
        (void *)&config.pcg_exit_tol,
		(void *)&config.empty_pinv
    };
    size_t ppcg_kernel_smem_size = pcgSharedMemSize<T>(STATE_SIZE, KNOT_POINTS);


    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

	gpuErrchk(cudaLaunchCooperativeKernel(pcg_kernel, KNOT_POINTS, NUM_THREADS, pcgKernelArgs, ppcg_kernel_smem_size));  
	gpuErrchk(cudaPeekAtLastError());
}

