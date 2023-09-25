#pragma once
#include <cstdint>
#include "gpuassert.cuh"
#include "glass.cuh"
#include "rbdfiles/rbd_plant.cuh"
#include "merit.cuh"
#include "matrix_utils.cuh"
#include "schur_inner.cuh"
#include "integrator.cuh"
#include "qdldl.h"
#include "csr.cuh"

template <typename T>
__global__
void gato_form_schur_jacobi(uint32_t state_size,
                            uint32_t control_size,
                            uint32_t knot_points,
                            T *d_G,
                            T *d_C,
                            T *d_g,
                            T *d_c,
                            T *d_S,
                            T *d_Pinv, 
                            T *d_gamma,
                            T rho,
                            uint32_t num_blocks)
{


    
    extern __shared__ T s_temp[ ];


    for(unsigned blockrow=blockIdx.x; blockrow<knot_points; blockrow+=num_blocks){


        gato_form_schur_jacobi_inner<T>(state_size, control_size, knot_points, d_G, d_C, d_g, d_c, d_S, d_Pinv, d_gamma, rho, s_temp, blockrow);
        // gato_form_schur_jacobi_inner(
        //     state_size,
        //     control_size,
        //     knot_points,
        //     d_G,
        //     d_C,
        //     d_g,
        //     d_c,
        //     d_S,
        //     d_Pinv,
        //     d_gamma,
        //     rho,
        //     s_temp,
        //     blockrow
        // );
    
    }
}


template <typename T>
__global__
void form_schur_qdl_kernel(uint32_t state_size,
                            uint32_t control_size,
                            uint32_t knot_points,
                            T *d_G,
                            T *d_C,
                            T *d_g,
                            T *d_c,
                            QDLDL_float *d_val,
                            T *d_gamma,
                            T rho)
{


    
    extern __shared__ T s_temp[ ];    
    const uint32_t states_sq = state_size*state_size;
    const uint32_t states_p_controls = state_size * control_size;
    const uint32_t controls_sq = control_size * control_size;
    const uint32_t states_s_controls = state_size + control_size;


    for(unsigned blockrow=blockIdx.x; blockrow<knot_points; blockrow+=gridDim.x){

        //  SPACE ALLOCATION IN SHARED MEM
        //  | phi_k | theta_k | thetaInv_k | gamma_k | block-specific...
        //     s^2      s^2         s^2         s
        T *s_phi_k = s_temp; 	                            	    // phi_k        states^2
        T *s_theta_k = s_phi_k + states_sq; 			            // theta_k      states^2
        T *s_thetaInv_k = s_theta_k + states_sq; 			        // thetaInv_k   states^2
        T *s_gamma_k = s_thetaInv_k + states_sq;                       // gamma_k      states
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

            gato_memcpy(s_Q0, d_G, states_sq);
            gato_memcpy(s_QN, d_G+(knot_points-1)*(states_sq+controls_sq), states_sq);
            gato_memcpy(s_q0, d_g, state_size);
            gato_memcpy(s_qN, d_g+(knot_points-1)*(state_size+control_size), state_size);

            __syncthreads();//----------------------------------------------------------------

            add_identity<T>(s_Q0, state_size, rho);
            add_identity<T>(s_QN, state_size, rho);
            
            __syncthreads();//----------------------------------------------------------------
            
            // SHARED MEMORY STATE
            // | Q_N | . | q_N | Q_0 | . | q_0 | scatch
            
            __syncthreads();//----------------------------------------------------------------


            // invert Q_N, Q_0
            loadIdentity<T>( state_size,state_size,s_Q0_i, s_QN_i);
            __syncthreads();//----------------------------------------------------------------
            invertMatrix<T>( state_size,state_size,state_size,s_Q0, s_QN, s_extra_temp);
            
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
            store_block_csr_lowertri<T>(state_size, knot_points, s_Q0_i, d_val, 1, blockrow, -1);

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


            const unsigned C_set_size = states_sq+states_p_controls;
            const unsigned G_set_size = states_sq+controls_sq;

            //  NON-LEADING BLOCK GOAL SHARED MEMORY STATE
            //  ...gamma_k | A_k | B_k | . | Q_k_I | . | Q_k+1_I | . | R_k_I | q_k | q_k+1 | r_k | integrator_error | extra_temp
            //               s^2   s*c  s^2   s^2   s^2    s^2    s^2   s^2     s      s      s          s                <s^2?

            T *s_Ak = s_end_main; 								
            T *s_Bk = s_Ak +        states_sq;
            T *s_Qk = s_Bk +        states_p_controls; 	
            T *s_Qk_i = s_Qk +      states_sq;	
            T *s_Qkp1 = s_Qk_i +    states_sq;
            T *s_Qkp1_i = s_Qkp1 +  states_sq;
            T *s_Rk = s_Qkp1_i +    states_sq;
            T *s_Rk_i = s_Rk +      controls_sq;
            T *s_qk = s_Rk_i +      controls_sq; 	
            T *s_qkp1 = s_qk +      state_size; 			
            T *s_rk = s_qkp1 +      state_size;
            T *s_end = s_rk +       control_size;
            
            // scratch
            T *s_extra_temp = s_end;
            

            __syncthreads();//----------------------------------------------------------------

            gato_memcpy(s_Ak,   d_C+      (blockrow-1)*C_set_size,                        states_sq);
            gato_memcpy(s_Bk,   d_C+      (blockrow-1)*C_set_size+states_sq,              states_p_controls);
            gato_memcpy(s_Qk,   d_G+      (blockrow-1)*G_set_size,                        states_sq);
            gato_memcpy(s_Qkp1, d_G+    (blockrow*G_set_size),                          states_sq);
            gato_memcpy(s_Rk,   d_G+      ((blockrow-1)*G_set_size+states_sq),            controls_sq);
            gato_memcpy(s_qk,   d_g+      (blockrow-1)*(states_s_controls),               state_size);
            gato_memcpy(s_qkp1, d_g+    (blockrow)*(states_s_controls),                 state_size);
            gato_memcpy(s_rk,   d_g+      ((blockrow-1)*(states_s_controls)+state_size),  control_size);

            __syncthreads();//----------------------------------------------------------------

            add_identity<T>(s_Qk, state_size, rho);
            add_identity<T>(s_Qkp1, state_size, rho);
            add_identity<T>(s_Rk, control_size, rho);
            
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
            gato_memcpy(
                d_G+(blockrow-1)*G_set_size,
                s_Qk_i,
                states_sq
            );

            // save Rk_i into G (now Ginv) for calculating dz
            gato_memcpy( 
                d_G+(blockrow-1)*G_set_size+states_sq,
                s_Rk_i,
                controls_sq
            );

            if(blockrow==knot_points-1){
                // save Qkp1_i into G (now Ginv) for calculating dz
                gato_memcpy(
                    d_G+(blockrow)*G_set_size,
                    s_Qkp1_i,
                    states_sq
                );
            }
            __syncthreads();//----------------------------------------------------------------

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


            for(unsigned i = threadIdx.x; i < states_sq; i += blockDim.x){
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

            for(unsigned i = threadIdx.x; i < states_sq; i += blockDim.x){
                s_theta_k[i] += s_Qkp1_i[i];
            }
            __syncthreads();//----------------------------------------------------------------

            // // save phi_k into left off-diagonal of S, 
            store_block_csr_lowertri<T>(state_size, knot_points, s_phi_k, d_val, 0, blockrow, -1);
            
            __syncthreads();//----------------------------------------------------------------


            // save -s_theta_k main diagonal S
            store_block_csr_lowertri<T>(state_size, knot_points, s_theta_k, d_val, 1, blockrow, -1);
            
            __syncthreads();//----------------------------------------------------------------

            // save gamma_k in gamma
            for(unsigned ind = threadIdx.x; ind < state_size; ind += blockDim.x){
                unsigned offset = (blockrow)*state_size + ind;
                d_gamma[offset] = s_gamma_k[ind]*-1;
            }

            __syncthreads();//----------------------------------------------------------------

        }
        
    }
}


/*******************************************************************************
 *                           Interface Functions                                *
 *******************************************************************************/


template <typename T, unsigned INTEGRATOR_TYPE = 0, bool ANGLE_WRAP = false>
__global__
void gato_form_kkt(uint32_t state_size, uint32_t control_size, uint32_t knot_points,
                   T *d_G_dense, T *d_C_dense, T *d_g, T *d_c,
                   void *d_dynMem_const, T timestep,
                   T *d_eePos_traj, T *d_xs, T *d_xu)
{

    const cgrps::thread_block block = cgrps::this_thread_block();
    const uint32_t thread_id = threadIdx.x;
    const uint32_t num_threads = blockDim.x;
    const uint32_t block_id = blockIdx.x;
    const uint32_t num_blocks = gridDim.x;

    const uint32_t states_sq = state_size*state_size;
    const uint32_t states_p_controls = state_size * control_size;
    const uint32_t controls_sq = control_size * control_size;
    const uint32_t states_s_controls = state_size + control_size;
    

    extern __shared__ T s_temp[];

    T *s_xux = s_temp;
    T *s_eePos_traj = s_xux + 2*state_size + control_size;
    T *s_Qk = s_eePos_traj + 6;
    T *s_Rk = s_Qk + states_sq;
    T *s_qk = s_Rk + controls_sq;
    T *s_rk = s_qk + state_size;
    T *s_end = s_rk + control_size;

    
    for(unsigned k = block_id; k < knot_points-1; k += num_blocks){

        glass::copy<T>(2*state_size + control_size, &d_xu[k*states_s_controls], s_xux);
        glass::copy<T>(2 * 6, &d_eePos_traj[k*6], s_eePos_traj);
        
        __syncthreads();    

        if(k==knot_points-2){          // last block

            T *s_Ak = s_end;
            T *s_Bk = s_Ak + states_sq;
            T *s_Qkp1 = s_Bk + states_p_controls;
            T *s_qkp1 = s_Qkp1 + states_sq;
            T *s_integrator_error = s_qkp1 + state_size;
            T *s_extra_temp = s_integrator_error + state_size;
            
            integratorAndGradient<T, INTEGRATOR_TYPE, ANGLE_WRAP, true>(
                state_size, control_size,
                s_xux,
                s_Ak,
                s_Bk,
                s_integrator_error,
                s_extra_temp,
                d_dynMem_const,
                timestep,
                block
            );
            __syncthreads();
            
            gato_plant::trackingCostGradientAndHessian_lastblock<T>(
                state_size,
                control_size,
                s_xux,
                s_eePos_traj,
                s_Qk,
                s_qk,
                s_Rk,
                s_rk,
                s_Qkp1,
                s_qkp1,
                s_extra_temp,
                d_dynMem_const
            );
            __syncthreads();

            for(int i = thread_id; i < state_size; i+=num_threads){
                d_c[i] = d_xu[i] - d_xs[i];
            }
            glass::copy<T>(states_sq, s_Qk, &d_G_dense[(states_sq+controls_sq)*k]);
            glass::copy<T>(controls_sq, s_Rk, &d_G_dense[(states_sq+controls_sq)*k+states_sq]);
            glass::copy<T>(states_sq, s_Qkp1, &d_G_dense[(states_sq+controls_sq)*(k+1)]);
            glass::copy<T>(state_size, s_qk, &d_g[states_s_controls*k]);
            glass::copy<T>(control_size, s_rk, &d_g[states_s_controls*k+state_size]);
            glass::copy<T>(state_size, s_qkp1, &d_g[states_s_controls*(k+1)]);
            glass::copy<T>(states_sq, static_cast<T>(-1), s_Ak, &d_C_dense[(states_sq+states_p_controls)*k]);
            glass::copy<T>(states_p_controls, static_cast<T>(-1), s_Bk, &d_C_dense[(states_sq+states_p_controls)*k+states_sq]);
            glass::copy<T>(state_size, s_integrator_error, &d_c[state_size*(k+1)]);

        }
        else{                               // not last knot

            T *s_Ak = s_end;
            T *s_Bk = s_Ak + states_sq;
            T *s_integrator_error = s_Bk + states_p_controls;
            T *s_extra_temp = s_integrator_error + state_size;

            integratorAndGradient<T, 
                                  INTEGRATOR_TYPE, 
                                  ANGLE_WRAP, 
                                  true>
                                 (state_size, control_size,
                                  s_xux,
                                  s_Ak,
                                  s_Bk,
                                  s_integrator_error,
                                  s_extra_temp,
                                  d_dynMem_const,
                                  timestep,
                                  block);
            __syncthreads();
           
            gato_plant::trackingCostGradientAndHessian<T>(state_size,
                                                  control_size,
                                                  s_xux,
                                                  s_eePos_traj,
                                                  s_Qk,
                                                  s_qk,
                                                  s_Rk,
                                                  s_rk,
                                                  s_extra_temp,
                                                  d_dynMem_const);
            __syncthreads();
 
            glass::copy<T>(states_sq, s_Qk, &d_G_dense[(states_sq+controls_sq)*k]);
            glass::copy<T>(controls_sq, s_Rk, &d_G_dense[(states_sq+controls_sq)*k+states_sq]);
            glass::copy<T>(state_size, s_qk, &d_g[states_s_controls*k]);
            glass::copy<T>(control_size, s_rk, &d_g[states_s_controls*k+state_size]);
            glass::copy<T>(states_sq, static_cast<T>(-1), s_Ak, &d_C_dense[(states_sq+states_p_controls)*k]);
            glass::copy<T>(states_p_controls, static_cast<T>(-1), s_Bk, &d_C_dense[(states_sq+states_p_controls)*k+states_sq]);
            glass::copy<T>(state_size, s_integrator_error, &d_c[state_size*(k+1)]);
        }
    }
}



template <typename T>
void form_schur_qdl(uint32_t state_size, uint32_t control_size, uint32_t knot_points,
                T *d_G_dense, T *d_C_dense, T *d_g, T *d_c, 
                QDLDL_float *d_val, T *d_gamma,
                T rho)
{
    const uint32_t s_temp_size =sizeof(T)*(8 * state_size*state_size+   
                                7 * state_size+ 
                                state_size * control_size+
                                3 * control_size + 2 * control_size * control_size + 3);

    // form Schur, Pinv
    form_schur_qdl_kernel<T><<<knot_points, SCHUR_THREADS, s_temp_size>>>(state_size, control_size, knot_points, d_G_dense, d_C_dense, d_g, d_c, d_val, d_gamma, rho);
    
}


template <typename T>
void form_schur(uint32_t state_size, uint32_t control_size, uint32_t knot_points,
                T *d_G_dense, T *d_C_dense, T *d_g, T *d_c, 
                T *d_S, T *d_Pinv, T *d_gamma, 
                T rho)
{
    const uint32_t s_temp_size =sizeof(T)*(8 * state_size*state_size+   
                                7 * state_size+ 
                                state_size * control_size+
                                3 * control_size + 2 * control_size * control_size + 3);

    // form Schur, Pinv
    gato_form_schur_jacobi<T><<<knot_points, SCHUR_THREADS, s_temp_size>>>(state_size, control_size, knot_points, d_G_dense, d_C_dense, d_g, d_c, d_S, d_Pinv, d_gamma, rho, knot_points);// hard coded
    
}


template <typename T>
void compute_dz(uint32_t state_size, uint32_t control_size, uint32_t knot_points, T *d_G_dense, T *d_C_dense, T *d_g_val, T *d_lambda, T *d_dz){
    
    compute_dz_kernel<<<knot_points, DZ_THREADS, sizeof(T)*(2*state_size*state_size+state_size)>>>(state_size, control_size, knot_points, d_G_dense, d_C_dense, d_g_val, d_lambda, d_dz);
}