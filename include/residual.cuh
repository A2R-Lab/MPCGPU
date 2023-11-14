#include <thrust/extrema.h>
#include <thrust/execution_policy.h>

struct max_abs_value
{
	template <typename T>
  __device__
  bool operator()(T lhs, T rhs)
  {
    return abs(lhs) < abs(rhs);
  }
};

template <typename T>
__global__
 void res_kernel(T * d_A, T * d_H, T * d_g,  T * d_x, T * d_lambda, T *d_z, T * d_Ax, T * d_Axz,  T * d_Hx, T * d_Atl, T * d_res){
	/* ||Ax - z||*/

	/* Ax = A * x */
	glass::gemv<T, false>(NC, NX, 1, d_A, d_x, d_Ax);

	/* Axz = Ax - z */
	glass::axpby<T>(NC, 1, d_Ax, -1, d_z, d_Axz);
	/* || H*x + g + A.T*lamb ||*/

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
 void res(T * d_A, T * d_H, T * d_g,  T * d_x, T * d_lambda, T *d_z, T * h_primal_res, T * h_dual_res){
	/* Alloc d_Ax, d_Axz*/
	T *d_Ax, *d_Axz;
	gpuErrchk(cudaMalloc(&d_Ax, NC * sizeof(T)));
	gpuErrchk(cudaMalloc(&d_Axz, NC * sizeof(T)));

	T *d_Hx, *d_Atl, *d_res;
	gpuErrchk(cudaMalloc(&d_Hx, NX * sizeof(T)));
	gpuErrchk(cudaMalloc(&d_Atl, NX * sizeof(T)));
	gpuErrchk(cudaMalloc(&d_res, NX * sizeof(T)));

	/* Launch */
	void *res_kernel_ptr = (void *) res_kernel<T>;

	void *kernelArgs[] = {
		(void *)&d_A, 
		(void *)&d_H,
		(void *)&d_g,
		(void *)&d_x,
		(void *)&d_lambda,
		(void *)&d_z, 
		(void *)&d_Ax, 
		(void *)&d_Axz,
		(void *)&d_Hx, 
		(void *)&d_Atl,
		(void *)&d_res
	};

	gpuErrchk(cudaLaunchCooperativeKernel(res_kernel_ptr, 1, NUM_THREADS, kernelArgs, 0));    
    gpuErrchk(cudaPeekAtLastError());
	/* Launch*/

	/* prima_res = norm(primal_res) */
	T * d_primal_res = thrust::max_element(thrust::device, d_Axz, d_Axz + NC, max_abs_value());
	gpuErrchk(cudaMemcpy(h_primal_res, d_primal_res,  sizeof(T), cudaMemcpyDeviceToHost));


	/* dual_res = norm(dual_res) */
	T * d_dual_res = thrust::max_element(thrust::device, d_res, d_res + NX, max_abs_value());
	gpuErrchk(cudaMemcpy(h_dual_res, d_dual_res,  sizeof(T), cudaMemcpyDeviceToHost));

	gpuErrchk(cudaFree(d_Ax));
	gpuErrchk(cudaFree(d_Axz));
	gpuErrchk(cudaFree(d_Hx));
	gpuErrchk(cudaFree(d_Atl));
	gpuErrchk(cudaFree(d_res));
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

	gpuErrchk(cudaLaunchCooperativeKernel(primal_res_kernel_ptr, 1, NUM_THREADS, kernelArgs, 0));    
    gpuErrchk(cudaPeekAtLastError());
	/* Launch*/

	/* prima_res = norm(primal_res) */
	T * d_abs_max = thrust::max_element(thrust::device, d_Axz, d_Axz + NC, max_abs_value());
	
	T h_abs_max[1];
	gpuErrchk(cudaMemcpy(h_abs_max, d_abs_max,  sizeof(T), cudaMemcpyDeviceToHost));

	gpuErrchk(cudaFree(d_Ax));
	gpuErrchk(cudaFree(d_Axz));

	return abs(*h_abs_max);

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

	gpuErrchk(cudaLaunchCooperativeKernel(dual_res_kernel_ptr, 1, NUM_THREADS, kernelArgs, 0));    
    gpuErrchk(cudaPeekAtLastError());

	/* dual_res = norm(dual_res) */
	T * d_abs_max = thrust::max_element(thrust::device, d_res, d_res + NX, max_abs_value());
	
	T h_abs_max[1];
	gpuErrchk(cudaMemcpy(h_abs_max, d_abs_max,  sizeof(T), cudaMemcpyDeviceToHost));


	gpuErrchk(cudaFree(d_Hx));
	gpuErrchk(cudaFree(d_Atl));
	gpuErrchk(cudaFree(d_res));

	return abs(*h_abs_max);
}