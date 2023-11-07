
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