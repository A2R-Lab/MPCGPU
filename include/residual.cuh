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