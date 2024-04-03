#pragma once
#include <cstdint>
#include "gpuassert.cuh"
#include "glass.cuh"
#include "utils/matrix.cuh"



template <typename T>
__device__
void complete_SS_Pinv_blockrow(uint32_t state_size, uint32_t knot_points, T *d_S, T *d_Pinv, T *d_gamma, T *s_temp, unsigned blockrow){

    const uint32_t states_sq = state_size*state_size;
    
    //  STATE OF DEVICE MEM
    //  S:      -Q0_i in spot 00, phik left off-diagonal, thetak main diagonal, phik_T right off-diagonal
    //  Phi:    -Q0 in spot 00, theta_invk main diagonal
    //  gamma:  -Q0_i*q0 spot 0, gammak


    // GOAL SPACE ALLOCATION IN SHARED MEM
    // s_temp  = | phi_k_T | phi_k | phi_kp1 | thetaInv_k | thetaInv_kp1 | thetaInv_km1 | PhiInv_R | PhiInv_L | scratch
    T *s_phi_k = s_temp;
    T *s_phi_kp1_T = s_phi_k + states_sq;
    T *s_thetaInv_k = s_phi_kp1_T + states_sq;
    T *s_thetaInv_km1 = s_thetaInv_k + states_sq;
    T *s_thetaInv_kp1 = s_thetaInv_km1 + states_sq;
    T *s_PhiInv_k_R = s_thetaInv_kp1 + states_sq;
    T *s_PhiInv_k_L = s_PhiInv_k_R + states_sq;
    T *s_scratch = s_PhiInv_k_L + states_sq;

    const unsigned lastrow = knot_points - 1;

    // load phi_kp1_T
    if(blockrow!=lastrow){
        load_block_bd<T>(
            state_size, knot_points,
            d_S,                // src
            s_phi_kp1_T,        // dst
            0,                  // block column (0, 1, or 2)
            blockrow+1,          // block row
            true                // transpose
        );
    }
    

    // load phi_k
    if(blockrow!=0){
        load_block_bd<T>(
            state_size,
            knot_points,
            d_S,
            s_phi_k,
            0,
            blockrow
        );
    }
    


    // load thetaInv_k
    load_block_bd<T>(
        state_size, knot_points,
        d_Pinv,
        s_thetaInv_k,
        1,
        blockrow
    );


    // load thetaInv_km1
    if(blockrow!=0){
        load_block_bd<T>(
            state_size, knot_points,
            d_Pinv,
            s_thetaInv_km1,
            1,
            blockrow-1
        );
    }


    // load thetaInv_kp1
    if(blockrow!=lastrow){
        load_block_bd<T>(
            state_size, knot_points,
            d_Pinv,
            s_thetaInv_kp1,
            1,
            blockrow+1
        );
    }
    

    __syncthreads();//----------------------------------------------------------------

    if(blockrow!=0){

        // compute left off diag    
        glass::gemm<T>(state_size, state_size, state_size                           , static_cast<T>(1.0), s_thetaInv_k, s_phi_k, s_scratch);
        __syncthreads();//----------------------------------------------------------------
        glass::gemm<T>(state_size, state_size, state_size, static_cast<T>(1.0), s_scratch, s_thetaInv_km1, s_PhiInv_k_L);
        __syncthreads();//----------------------------------------------------------------

        // store left diagonal in Phi
        store_block_bd<T>(
            state_size, knot_points,
            s_PhiInv_k_L, 
            d_Pinv,
            0,
            blockrow,
            -1
        );
        __syncthreads();//----------------------------------------------------------------
    }


    if(blockrow!=lastrow){

        // calculate Phi right diag
        glass::gemm<T>(state_size, state_size, state_size                           , static_cast<T>(1.0), s_thetaInv_k, s_phi_kp1_T, s_scratch);
        __syncthreads();//----------------------------------------------------------------
        glass::gemm<T>(state_size, state_size, state_size, static_cast<T>(1.0), s_scratch, s_thetaInv_kp1, s_PhiInv_k_R);
        __syncthreads();//----------------------------------------------------------------

        // store Phi right diag
        store_block_bd<T>(
            state_size, knot_points,
            s_PhiInv_k_R, 
            d_Pinv,
            2,
            blockrow,
            -1
        );

    }
}

template <typename T>
__device__
void form_S_gamma_and_jacobi_Pinv_blockrow(uint32_t state_size, uint32_t control_size, uint32_t knot_points, T *d_G, T *d_C, T *d_g, T *d_c, T *d_S, T *d_Pinv, T *d_gamma, T rho, T *s_temp, unsigned blockrow){
    
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

        glass::copy<T>(state_size*state_size, d_G, s_Q0);
        glass::copy<T>(state_size*state_size, d_G+(knot_points-1)*(state_size*state_size+control_size*control_size), s_QN);
        glass::copy<T>(state_size, d_g, s_q0);
        glass::copy<T>(state_size, d_g+(knot_points-1)*(state_size+control_size), s_qN);

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

        glass::copy<T>(state_size*state_size, d_C+      (blockrow-1)*C_set_size, s_Ak);
        glass::copy<T>(state_size*control_size, d_C+      (blockrow-1)*C_set_size+state_size*state_size, s_Bk);
        glass::copy<T>(state_size*state_size, d_G+      (blockrow-1)*G_set_size, s_Qk);
        glass::copy<T>(state_size*state_size, d_G+    (blockrow*G_set_size), s_Qkp1);
        glass::copy<T>(control_size*control_size, d_G+      ((blockrow-1)*G_set_size+state_size*state_size), s_Rk);
        glass::copy<T>(state_size, d_g+      (blockrow-1)*(state_size+control_size), s_qk);
        glass::copy<T>(state_size, d_g+    (blockrow)*(state_size+control_size), s_qkp1);
        glass::copy<T>(control_size, d_g+      ((blockrow-1)*(state_size+control_size)+state_size), s_rk);

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
        glass::copy<T>(state_size*state_size, s_Qk_i, d_G+(blockrow-1)*G_set_size);

        // save Rk_i into G (now Ginv) for calculating dz
        glass::copy<T>(control_size*control_size, s_Rk_i, d_G+(blockrow-1)*G_set_size+state_size*state_size);

        if(blockrow==knot_points-1){
            // save Qkp1_i into G (now Ginv) for calculating dz
            glass::copy<T>(state_size*state_size, s_Qkp1_i, d_G+(blockrow)*G_set_size);
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
        glass::gemm<T>(state_size, state_size, state_size, static_cast<T>(1.0), s_Ak, s_Qk_i, s_phi_k);
        // for(int i = threadIdx.x; i < state_size*state_size; i++){
        //     s_phi_k[i] *= -1;
        // }

        __syncthreads();//----------------------------------------------------------------

        // Compute -BR^{-1} in Qkp1
        glass::gemm<T>(state_size, control_size, control_size, static_cast<T>(1.0), s_Bk, s_Rk_i, s_Qkp1);

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
        glass::gemm<T, true>(
            state_size, 
            state_size, 
            state_size,
            static_cast<T>(1.0), 
            s_phi_k, 
            s_Ak, 
            s_theta_k
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
        glass::gemm<T, true>(
            state_size,
            control_size,
            state_size,
            static_cast<T>(1.0),
            s_Qkp1,
            s_Bk,
            s_Qkp1_i
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
        glass::gemm<T, true>(
            state_size, 
            state_size, 
            state_size,
            static_cast<T>(1.0), 
            s_Ak, 
            s_phi_k, 
            s_Qkp1
        );
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
__global__
void form_S_gamma_Pinv_kernel(
    uint32_t state_size,
    uint32_t control_size,
    uint32_t knot_points,
    T *d_G,
    T *d_C,
    T *d_g,
    T *d_c,
    T *d_S,
    T *d_Pinv, 
    T *d_gamma,
    T rho
){

    extern __shared__ T s_temp[ ];

    for(unsigned blockrow=blockIdx.x; blockrow<knot_points; blockrow+=gridDim.x){
        form_S_gamma_and_jacobi_Pinv_blockrow<T>(
            state_size, 
            control_size, 
            knot_points, 
            d_G, 
            d_C, 
            d_g, 
            d_c, 
            d_S, 
            d_Pinv, 
            d_gamma, 
            rho, 
            s_temp, 
            blockrow
        );
    }
    cgrps::this_grid().sync();

    for(unsigned blockrow=blockIdx.x; blockrow<knot_points; blockrow+=gridDim.x){
        complete_SS_Pinv_blockrow<T>(
            state_size, knot_points,
            d_S,
            d_Pinv,
            d_gamma,
            s_temp,
            blockrow
        );
    }
}


/*******************************************************************************
 *                           Interface Functions                                *
 *******************************************************************************/


template <typename T>
void form_schur_system(
    uint32_t state_size, 
    uint32_t control_size, 
    uint32_t knot_points,
    T *d_G_dense, 
    T *d_C_dense, 
    T *d_g, 
    T *d_c, 
    T *d_S, 
    T *d_Pinv, 
    T *d_gamma,            
    T rho
){
    const uint32_t s_temp_size = sizeof(T)*(8 * state_size*state_size +
                                            7 * state_size + 
                                            state_size * control_size +
                                            3 * control_size + 
                                            2 * control_size * control_size + 
                                            3);

    void *kernel = (void *) form_S_gamma_Pinv_kernel<T>;
    void *args[] = {
        (void *) &state_size,
        (void *) &control_size,
        (void *) &knot_points,
        (void *) &d_G_dense,
        (void *) &d_C_dense,
        (void *) &d_g,
        (void *) &d_c,
        (void *) &d_S,
        (void *) &d_Pinv,
        (void *) &d_gamma,
        (void *) &rho
    };

    gpuErrchk(cudaLaunchCooperativeKernel(kernel, knot_points, SCHUR_THREADS, args, s_temp_size));
}