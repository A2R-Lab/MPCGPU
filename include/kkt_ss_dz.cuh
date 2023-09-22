#pragma once
#include <cstdint>

    //TODO: this
    template <typename T>
    size_t get_kkt_smem_size(uint32_t state_size, uint32_t control_size){
        // const uint32_t states_p_controls = state_size * control_size;
        const uint32_t states_sq = state_size * state_size;
        const uint32_t controls_sq = control_size * control_size;
        ///TODO: costGradientAndHessian_TempMemSize_Shared < costAndGradient_TempMemSize_Shared ? seems odd
        // bad changed
        // return sizeof(T)*(3*state_size*state_size + control_size*control_size + 6 * state_size + 2 * control_size + states_p_controls + (state_size/2)*(state_size + control_size + 1) + gato_plant::forwardDynamicsAndGradient_TempMemSize_Shared());
        
        // original 
        // return sizeof(float)*(3*states_sq + controls_sq + 6 * state_size + 2 * control_size + state_size*control_size + max(gato_plant::costAndGradient_TempMemSize_Shared(), (state_size/2)*(state_size + control_size + 1) + gato_plant::forwardDynamicsAndGradient_TempMemSize_Shared()));

        // condensed
        return sizeof(T)*(3*states_sq + controls_sq + 7 * state_size + 3 * control_size + state_size*control_size + max(grid::EE_POS_SHARED_MEM_COUNT, grid::DEE_POS_SHARED_MEM_COUNT) + max((state_size/2)*(state_size + control_size + 1) + gato_plant::forwardDynamicsAndGradient_TempMemSize_Shared(), 3 + (state_size/2)*6));
        // same but not condensed
        // return sizeof(T)*(5*state_size + 3*control_size + states_sq + controls_sq + 2 * states_sq + state_size*control_size + 2*state_size + (state_size/2)*(state_size + control_size + 1) + gato_plant::forwardDynamicsAndGradient_TempMemSize_Shared());
    }



    template <typename T>
    __device__
    void gato_form_schur_jacobi_inner(uint32_t state_size, uint32_t control_size, uint32_t knot_points, T *d_G, T *d_C, T *d_g, T *d_c, T *d_S, T *d_Pinv, T *d_gamma, T rho, T *s_temp, unsigned blockrow){
        
        //  SPACE ALLOCATION IN SHARED MEM
        //  | phi_k | theta_k | thetaInv_k | gamma_k | block-specific...
        //     s^2      s^2         s^2         s
        T *s_phi_k = s_temp; 	                            	    // phi_k        states^2
        T *s_theta_k = s_phi_k + state_size*state_size; 			            // theta_k      states^2
        T *s_thetaInv_k = s_theta_k + state_size*state_size; 			        // thetaInv_k   states^2
        T *s_gamma_k = s_thetaInv_k + state_size*state_size;                       // gamma_k      states
        T *s_end_main = s_gamma_k + state_size;                               

        if(blockrow==0){

            //  LEADING BLOCK GOAL SHARED MEMORY STATE
            //  ...gamma_k | . | Q_N_I | q_N | . | Q_0_I | q_0 | scatch
            //              s^2   s^2     s   s^2   s^2     s      ? 
        
            T *s_QN = s_end_main;
            T *s_QN_i = s_QN + state_size * state_size;
            T *s_qN = s_QN_i + state_size * state_size;
            T *s_Q0 = s_qN + state_size;
            T *s_Q0_i = s_Q0 + state_size * state_size;
            T *s_q0 = s_Q0_i + state_size * state_size;
            T *s_end = s_q0 + state_size;

            // scratch space
            T *s_R_not_needed = s_end;
            T *s_r_not_needed = s_R_not_needed + control_size * control_size;
            T *s_extra_temp = s_r_not_needed + control_size * control_size;

            __syncthreads();//----------------------------------------------------------------

            gato_memcpy<T>(s_Q0, d_G, state_size*state_size);
            gato_memcpy<T>(s_QN, d_G+(knot_points-1)*(state_size*state_size+control_size*control_size), state_size*state_size);
            gato_memcpy<T>(s_q0, d_g, state_size);
            gato_memcpy<T>(s_qN, d_g+(knot_points-1)*(state_size+control_size), state_size);

            __syncthreads();//----------------------------------------------------------------

            add_identity(s_Q0, state_size, rho);
            add_identity(s_QN, state_size, rho);
            // if(PRINT_THREAD){
            //     printf("Q0\n");
            //     printMat<state_size,state_size>(s_Q0,state_size);
            //     printf("q0\n");
            //     printMat<1,state_size>(s_q0,1);
            //     printf("QN\n");
            //     printMat<state_size,state_size>(s_QN,state_size);
            //     printf("qN\n");
            //     printMat<1,state_size>(s_qN,1);
            //     printf("start error\n");
            //     printMat<1,state_size>(s_integrator_error,1);
            //     printf("\n");
            // }
            __syncthreads();//----------------------------------------------------------------
            
            // SHARED MEMORY STATE
            // | Q_N | . | q_N | Q_0 | . | q_0 | scatch
            

            // save -Q_0 in PhiInv spot 00
            store_block_bd<T>(
                state_size,
                knot_points,
                s_Q0,                       // src     
                d_Pinv,                   // dst         
                1,                          // col
                blockrow,                    // blockrow
                -1                          //  multiplier
            );
            __syncthreads();//----------------------------------------------------------------


            // invert Q_N, Q_0
            loadIdentity<T>( state_size,state_size,s_Q0_i, s_QN_i);
            __syncthreads();//----------------------------------------------------------------
            invertMatrix<T>( state_size,state_size,state_size,s_Q0, s_QN, s_extra_temp);
            
            __syncthreads();//----------------------------------------------------------------


            // if(PRINT_THREAD){
            //     printf("Q0Inv\n");
            //     printMat<state_size,state_size>(s_Q0_i,state_size);
            //     printf("QNInv\n");
            //     printMat<floatstate_size,state_size>(s_QN_i,state_size);
            //     printf("theta\n");
            //     printMat<floatstate_size,state_size>(s_theta_k,state_size);
            //     printf("thetaInv\n");
            //     printMat<floatstate_size,state_size>(s_thetaInv_k,state_size);
            //     printf("\n");
            // }
            __syncthreads();//----------------------------------------------------------------

            // SHARED MEMORY STATE
            // | . | Q_N_i | q_N | . | Q_0_i | q_0 | scatch
            

            // compute gamma
            mat_vec_prod<T>( state_size, state_size,
                s_Q0_i,                                    
                s_q0,                                       
                s_gamma_k 
            );
            __syncthreads();//----------------------------------------------------------------
            

            // save -Q0_i in spot 00 in S
            store_block_bd<T>( state_size, knot_points,
                s_Q0_i,                         // src             
                d_S,                            // dst              
                1,                              // col   
                blockrow,                        // blockrow         
                -1                              //  multiplier   
            );
            __syncthreads();//----------------------------------------------------------------


            // compute Q0^{-1}q0
            mat_vec_prod<T>( state_size, state_size,
                s_Q0_i,
                s_q0,
                s_Q0
            );
            __syncthreads();//----------------------------------------------------------------


            // SHARED MEMORY STATE
            // | . | Q_N_i | q_N | Q0^{-1}q0 | Q_0_i | q_0 | scatch


            // save -Q0^{-1}q0 in spot 0 in gamma
            for(unsigned ind = threadIdx.x; ind < state_size; ind += blockDim.x){
                d_gamma[ind] = -s_Q0[ind];
            }
            __syncthreads();//----------------------------------------------------------------

        }
        else{                       // blockrow!=LEAD_BLOCK


            const unsigned C_set_size = state_size*state_size+state_size*control_size;
            const unsigned G_set_size = state_size*state_size+control_size*control_size;

            //  NON-LEADING BLOCK GOAL SHARED MEMORY STATE
            //  ...gamma_k | A_k | B_k | . | Q_k_I | . | Q_k+1_I | . | R_k_I | q_k | q_k+1 | r_k | integrator_error | extra_temp
            //               s^2   s*c  s^2   s^2   s^2    s^2    s^2   s^2     s      s      s          s                <s^2?

            T *s_Ak = s_end_main; 								
            T *s_Bk = s_Ak +        state_size*state_size;
            T *s_Qk = s_Bk +        state_size*control_size; 	
            T *s_Qk_i = s_Qk +      state_size*state_size;	
            T *s_Qkp1 = s_Qk_i +    state_size*state_size;
            T *s_Qkp1_i = s_Qkp1 +  state_size*state_size;
            T *s_Rk = s_Qkp1_i +    state_size*state_size;
            T *s_Rk_i = s_Rk +      control_size*control_size;
            T *s_qk = s_Rk_i +      control_size*control_size; 	
            T *s_qkp1 = s_qk +      state_size; 			
            T *s_rk = s_qkp1 +      state_size;
            T *s_end = s_rk +       control_size;
            
            // scratch
            T *s_extra_temp = s_end;
            

            // if(PRINT_THREAD){
            //     printf("xk\n");
            //     printMat<float1,state_size>(s_xux,1);
            //     printf("uk\n");
            //     printMat<float1,control_size>(&s_xux[state_size],1);
            //     printf("xkp1\n");
            //     printMat<float1,state_size>(&s_xux[state_size+control_size],1);
            //     printf("\n");
            // }

            __syncthreads();//----------------------------------------------------------------

            gato_memcpy<T>(s_Ak,   d_C+      (blockrow-1)*C_set_size,                        state_size*state_size);
            gato_memcpy<T>(s_Bk,   d_C+      (blockrow-1)*C_set_size+state_size*state_size,              state_size*control_size);
            gato_memcpy<T>(s_Qk,   d_G+      (blockrow-1)*G_set_size,                        state_size*state_size);
            gato_memcpy<T>(s_Qkp1, d_G+    (blockrow*G_set_size),                          state_size*state_size);
            gato_memcpy<T>(s_Rk,   d_G+      ((blockrow-1)*G_set_size+state_size*state_size),            control_size*control_size);
            gato_memcpy<T>(s_qk,   d_g+      (blockrow-1)*(state_size+control_size),               state_size);
            gato_memcpy<T>(s_qkp1, d_g+    (blockrow)*(state_size+control_size),                 state_size);
            gato_memcpy<T>(s_rk,   d_g+      ((blockrow-1)*(state_size+control_size)+state_size),  control_size);

            __syncthreads();//----------------------------------------------------------------

            add_identity(s_Qk, state_size, rho);
            add_identity(s_Qkp1, state_size, rho);
            add_identity(s_Rk, control_size, rho);

    #if DEBUG_MODE    
            if(blockIdx.x==1 && threadIdx.x==0){
                printf("Ak\n");
                printMat<state_size,state_size>(s_Ak,state_size);
                printf("Bk\n");
                printMat<state_size,control_size>(s_Bk,state_size);
                printf("Qk\n");
                printMat<state_size,state_size>(s_Qk,state_size);
                printf("Rk\n");
                printMat<control_size,control_size>(s_Rk,control_size);
                printf("qk\n");
                printMat<state_size, 1>(s_qk,1);
                printf("rk\n");
                printMat<control_size, 1>(s_rk,1);
                printf("Qkp1\n");
                printMat<state_size,state_size>(s_Qkp1,state_size);
                printf("qkp1\n");
                printMat<state_size, 1>(s_qkp1,1);
                printf("integrator error\n");
            }
            __syncthreads();//----------------------------------------------------------------
    #endif /* #if DEBUG_MODE */
            
            // Invert Q, Qp1, R 
            loadIdentity<T>( state_size,state_size,control_size,
                s_Qk_i, 
                s_Qkp1_i, 
                s_Rk_i
            );
            __syncthreads();//----------------------------------------------------------------
            invertMatrix<T>( state_size,state_size,control_size,state_size,
                s_Qk, 
                s_Qkp1, 
                s_Rk, 
                s_extra_temp
            );
            __syncthreads();//----------------------------------------------------------------

            // save Qk_i into G (now Ginv) for calculating dz
            gato_memcpy<T>(
                d_G+(blockrow-1)*G_set_size,
                s_Qk_i,
                state_size*state_size
            );

            // save Rk_i into G (now Ginv) for calculating dz
            gato_memcpy<T>( 
                d_G+(blockrow-1)*G_set_size+state_size*state_size,
                s_Rk_i,
                control_size*control_size
            );

            if(blockrow==knot_points-1){
                // save Qkp1_i into G (now Ginv) for calculating dz
                gato_memcpy<T>(
                    d_G+(blockrow)*G_set_size,
                    s_Qkp1_i,
                    state_size*state_size
                );
            }
            __syncthreads();//----------------------------------------------------------------

    #if DEBUG_MODE
            if(blockrow==1&&threadIdx.x==0){
                printf("Qk\n");
                printMat< state_size,state_size>(s_Qk_i,state_size);
                printf("RkInv\n");
                printMat<control_size,control_size>(s_Rk_i,control_size);
                printf("Qkp1Inv\n");
                printMat< state_size,state_size>(s_Qkp1_i,state_size);
                printf("\n");
            }
            __syncthreads();//----------------------------------------------------------------
    #endif /* #if DEBUG_MODE */


            // Compute -AQ^{-1} in phi
            mat_mat_prod<T>(
                s_phi_k,
                s_Ak,
                s_Qk_i,
                state_size, 
                state_size, 
                state_size, 
                state_size
            );
            // for(int i = threadIdx.x; i < state_size*state_size; i++){
            //     s_phi_k[i] *= -1;
            // }

            __syncthreads();//----------------------------------------------------------------

            // Compute -BR^{-1} in Qkp1
            mat_mat_prod<T>(
                s_Qkp1,
                s_Bk,
                s_Rk_i,
                state_size,
                control_size,
                control_size,
                control_size
            );

            __syncthreads();//----------------------------------------------------------------

            // compute Q_{k+1}^{-1}q_{k+1} - IntegratorError in gamma
            mat_vec_prod<T>( state_size, state_size,
                s_Qkp1_i,
                s_qkp1,
                s_gamma_k
            );
            for(unsigned i = threadIdx.x; i < state_size; i += blockDim.x){
                s_gamma_k[i] -= d_c[(blockrow*state_size)+i];
            }
            __syncthreads();//----------------------------------------------------------------

            // compute -AQ^{-1}q for gamma         temp storage in extra temp
            mat_vec_prod<T>( state_size, state_size,
                s_phi_k,
                s_qk,
                s_extra_temp
            );
            

            __syncthreads();//----------------------------------------------------------------
            
            // compute -BR^{-1}r for gamma           temp storage in extra temp + states
            mat_vec_prod<T>( state_size, control_size,
                s_Qkp1,
                s_rk,
                s_extra_temp + state_size
            );

            __syncthreads();//----------------------------------------------------------------
            
            // gamma = yeah...
            for(unsigned i = threadIdx.x; i < state_size; i += blockDim.x){
                s_gamma_k[i] += s_extra_temp[state_size + i] + s_extra_temp[i]; 
            }
            __syncthreads();//----------------------------------------------------------------

            // compute AQ^{-1}AT   -   Qkp1^{-1} for theta
            mat_mat_prod<T>(
                s_theta_k,
                s_phi_k,
                s_Ak,
                state_size,
                state_size,
                state_size,
                state_size,
                true
            );

            __syncthreads();//----------------------------------------------------------------

    #if DEBUG_MODE
            if(blockrow==1&&threadIdx.x==0){
                printf("this is the A thing\n");
                printMat< state_size, state_size>(s_theta_k, 234);
            }
    #endif /* #if DEBUG_MODE */

            for(unsigned i = threadIdx.x; i < state_size*state_size; i += blockDim.x){
                s_theta_k[i] += s_Qkp1_i[i];
            }
            
            __syncthreads();//----------------------------------------------------------------

            // compute BR^{-1}BT for theta            temp storage in QKp1{-1}
            mat_mat_prod<T>(
                s_Qkp1_i,
                s_Qkp1,
                s_Bk,
                state_size,
                control_size,
                state_size,
                control_size,
                true
            );

            __syncthreads();//----------------------------------------------------------------

            for(unsigned i = threadIdx.x; i < state_size*state_size; i += blockDim.x){
                s_theta_k[i] += s_Qkp1_i[i];
            }
            __syncthreads();//----------------------------------------------------------------

            // save phi_k into left off-diagonal of S, 
            store_block_bd<T>( state_size, knot_points,
                s_phi_k,                        // src             
                d_S,                            // dst             
                0,                              // col
                blockrow,                        // blockrow    
                -1
            );
            __syncthreads();//----------------------------------------------------------------

            // save -s_theta_k main diagonal S
            store_block_bd<T>( state_size, knot_points,
                s_theta_k,                                               
                d_S,                                                 
                1,                                               
                blockrow,
                -1                                             
            );          
            __syncthreads();//----------------------------------------------------------------

            // invert theta
            loadIdentity<T>(state_size,s_thetaInv_k);
            __syncthreads();//----------------------------------------------------------------
            invertMatrix<T>(state_size,s_theta_k, s_extra_temp);
            __syncthreads();//----------------------------------------------------------------


            // save thetaInv_k main diagonal PhiInv
            store_block_bd<T>( state_size, knot_points,
                s_thetaInv_k, 
                d_Pinv,
                1,
                blockrow,
                -1
            );

            __syncthreads();//----------------------------------------------------------------

            // save gamma_k in gamma
            for(unsigned ind = threadIdx.x; ind < state_size; ind += blockDim.x){
                unsigned offset = (blockrow)*state_size + ind;
                d_gamma[offset] = s_gamma_k[ind]*-1;
            }

            __syncthreads();//----------------------------------------------------------------

            //transpose phi_k
            loadIdentity<T>(state_size,s_Ak);
            __syncthreads();//----------------------------------------------------------------
            mat_mat_prod<T>(s_Qkp1,s_Ak,s_phi_k,state_size,state_size,state_size,state_size,true);
            __syncthreads();//----------------------------------------------------------------

            // save phi_k_T into right off-diagonal of S,
            store_block_bd<T>( state_size, knot_points,
                s_Qkp1,                        // src             
                d_S,                            // dst             
                2,                              // col
                blockrow-1,                      // blockrow    
                -1
            );

            __syncthreads();//----------------------------------------------------------------
        }

    }


    template <typename T>
    __device__
    void gato_compute_dz_inner(uint32_t state_size, uint32_t control_size, uint32_t knot_points, T *d_Ginv_dense, T *d_C_dense, T *d_g_val, T *d_lambda, T *d_dz, T *s_mem, int blockrow){

        const uint32_t states_sq = state_size*state_size;
        const uint32_t states_p_controls = state_size * control_size;
        const uint32_t controls_sq = control_size * control_size;
        const uint32_t states_s_controls = state_size + control_size;

        const unsigned set = blockrow/2;
        
        if(blockrow%2){ // control row
            // shared mem config
            //    Rkinv |   BkT
            //      C^2  |  S*C

            T *s_Rk_i = s_mem;
            T *s_BkT = s_Rk_i + controls_sq;
            T *s_scratch = s_BkT + states_p_controls;

            // load Rkinv from G
            gato_memcpy<T>(s_Rk_i, 
                        d_Ginv_dense+set*(states_sq+controls_sq)+states_sq, 
                        controls_sq);

            // load Bk from C
            gato_memcpy<T>(s_BkT,
                        d_C_dense+set*(states_sq+states_p_controls)+states_sq,
                        states_p_controls);

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
            gato_memcpy<T>(d_dz+set*(states_s_controls)+state_size,
                            s_scratch+control_size,
                            control_size);

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
            gato_memcpy<T>(s_Qk_i, 
                        d_Ginv_dense+set*(states_sq+controls_sq), 
                        states_sq);

                        ///TODO: linsys solver hasn't been checked with this change
            if(set != knot_points-1){
                // load Ak from C
                gato_memcpy<T>(s_AkT,
                    d_C_dense+set*(states_sq+states_p_controls),
                    states_sq);
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
                // cudaMemsetAsync(s_scratch, 0, state_size);       
                // need to compile with -dc flag to use deivce functions like that but having issues TODO: 
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
            gato_memcpy<T>(d_dz+set*(states_s_controls),
                            s_AkT,
                            state_size);
        }
    }

    template <typename T>
    __global__
    void compute_dz_kernel(uint32_t state_size, uint32_t control_size, uint32_t knot_points, T *d_G_dense, T *d_C_dense, T *d_g_val, T *d_lambda, T *d_dz){

        // const unsigned s_mem_size = max(2*control_size, state_size);

        extern __shared__ T s_mem[]; 

        for(int ind = blockIdx.x; ind < 2*knot_points-1; ind+=gridDim.x){
            gato_compute_dz_inner(state_size, control_size, knot_points, d_G_dense, d_C_dense, d_g_val, d_lambda, d_dz, s_mem, ind);
        }
    }



