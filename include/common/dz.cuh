#include "utils/matrix.cuh"

template <typename T>
__global__
void compute_dz_kernel(uint32_t state_size, uint32_t control_size, uint32_t knot_points, T *d_G_dense, T *d_C_dense, T *d_g_val, T *d_lambda, T *d_dz){

    extern __shared__ T s_mem[]; 
    
    const uint32_t states_sq = state_size*state_size;
    const uint32_t states_p_controls = state_size * control_size;
    const uint32_t controls_sq = control_size * control_size;
    const uint32_t states_s_controls = state_size + control_size;
    unsigned set;

    for(int blockrow = blockIdx.x; blockrow < 2*knot_points-1; blockrow+=gridDim.x){

        set = blockrow/2;
        
        if(blockrow%2){ // control row
            // shared mem config
            //    Rkinv |   BkT
            //      C^2  |  S*C

            T *s_Rk_i = s_mem;
            T *s_BkT = s_Rk_i + controls_sq;
            T *s_scratch = s_BkT + states_p_controls;

            // load Rkinv from G
            glass::copy<T>(controls_sq, d_G_dense+set*(states_sq+controls_sq)+states_sq, s_Rk_i);

            // load Bk from C
            glass::copy<T>(states_p_controls, d_C_dense+set*(states_sq+states_p_controls)+states_sq, s_BkT);

            __syncthreads();

            // // compute BkT*lkp1
            gato_ATx<T>(s_scratch,
                    s_BkT,
                    d_lambda+(set+1)*state_size,
                    state_size,
                    control_size);
            __syncthreads();

            // subtract from rk
            gato_vec_dif(s_scratch,
                        d_g_val+set*(states_s_controls)+state_size,
                        s_scratch,
                        control_size);
            __syncthreads();

            // multiply Rk_i*scratch in scratch + C
            mat_vec_prod<T>( control_size, control_size,s_Rk_i,
                                                            s_scratch,
                                                            s_scratch+control_size);
            __syncthreads();
            
            // store in d_dz
            glass::copy<T>(control_size, s_scratch+control_size, d_dz+set*(states_s_controls)+state_size);

        }
        else{   // state row

            T *s_Qk_i = s_mem;
            T *s_AkT = s_Qk_i + states_sq;
            T *s_scratch = s_AkT + states_sq;
            
            // shared mem config
            //    Qkinv |  AkT | scratch
            //      S^2     S^2

            /// TODO: error check
            // load Qkinv from G
            glass::copy<T>(states_sq, d_G_dense+set*(states_sq+controls_sq), s_Qk_i);

                        ///TODO: linsys solver hasn't been checked with this change
            if(set != knot_points-1){
                // load Ak from C
                glass::copy<T>(states_sq, d_C_dense+set*(states_sq+states_p_controls), s_AkT);
                __syncthreads();
                            
                // // compute AkT*lkp1 in scratch
                gato_ATx(s_scratch,
                        s_AkT,
                        d_lambda+(set+1)*state_size,
                        state_size,
                        state_size);
                __syncthreads();
            }
            else{
                for(int i = threadIdx.x; i < state_size; i+=blockDim.x){
                    s_scratch[i] = 0;
                }
            }
            

            // add lk to scratch
            gato_vec_sum<T>(s_scratch,     // out
                        d_lambda+set*state_size,
                        s_scratch,
                        state_size);
            __syncthreads();

            // subtract from qk in scratch
            gato_vec_dif<T>(s_scratch,
                        d_g_val+set*(states_s_controls),
                        s_scratch,
                        state_size);
            __syncthreads();
            
            
            // multiply Qk_i(scratch) in Akt
            mat_vec_prod<T>( state_size, state_size,s_Qk_i,
                                                        s_scratch,
                                                        s_AkT);
            __syncthreads();

            // store in dz
            glass::copy<T>(state_size, s_AkT, d_dz+set*(states_s_controls));
        }
    }
}


template <typename T>
void compute_dz(uint32_t state_size, uint32_t control_size, uint32_t knot_points, T *d_G_dense, T *d_C_dense, T *d_g_val, T *d_lambda, T *d_dz){
    
    compute_dz_kernel<<<knot_points, DZ_THREADS, sizeof(T)*(2*state_size*state_size+state_size)>>>(
        state_size, 
        control_size, 
        knot_points, 
        d_G_dense, 
        d_C_dense, 
        d_g_val, 
        d_lambda, 
        d_dz
    );
}