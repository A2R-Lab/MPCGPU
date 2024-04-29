
#include "kkt.cuh"

template <typename T, unsigned INTEGRATOR_TYPE = 0, bool ANGLE_WRAP = false>
__global__
void generate_kkt_submatrices_n(uint32_t solve_count,
							  uint32_t state_size, 
                              uint32_t control_size, 
                              uint32_t knot_points,
                              T *d_G_dense, 
                              T *d_C_dense, 
                              T *d_g, 
                              T *d_c,
                              void *d_dynMem_const, 
                              T timestep,
                              T *d_eePos_traj, 
                              T *d_xs, 
                              T *d_xu)
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
	const uint32_t traj_len = (state_size+control_size)*knot_points-control_size;
    

    extern __shared__ T s_temp[];

    T *s_xux = s_temp;
    T *s_eePos_traj = s_xux + 2*state_size + control_size;
    T *s_Qk = s_eePos_traj + 6;
    T *s_Rk = s_Qk + states_sq;
    T *s_qk = s_Rk + controls_sq;
    T *s_rk = s_qk + state_size;
    T *s_end = s_rk + control_size;


    for(unsigned b = block_id; b < solve_count*(knot_points-1); b += num_blocks){
		unsigned k = b / solve_count;
		unsigned prob = b % solve_count; 

		T *d_xu_i = d_xu + traj_len * prob;
		T *d_eePos_traj_i = d_eePos_traj + 6  * knot_points * prob;
		T *d_xs_i = d_xs + state_size * prob;
		T *d_G_dense_i = d_G_dense + ((states_sq+controls_sq)*knot_points-controls_sq) * prob;
		T *d_g_i = d_g + ((state_size+control_size)*knot_points-control_size) * prob;
		T *d_C_dense_i = d_C_dense + (states_sq+states_p_controls)*(knot_points-1)*prob;
		T *d_c_i = d_c + (state_size*knot_points) * prob;

        glass::copy<T>(2*state_size + control_size, &d_xu_i[k*states_s_controls], s_xux);
        glass::copy<T>(2 * 6, &d_eePos_traj_i[k*6], s_eePos_traj);
        
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
                d_c_i[i] = d_xu_i[i] - d_xs_i[i];
            }
            glass::copy<T>(states_sq, s_Qk, &d_G_dense_i[(states_sq+controls_sq)*k]);
            glass::copy<T>(controls_sq, s_Rk, &d_G_dense_i[(states_sq+controls_sq)*k+states_sq]);
            glass::copy<T>(states_sq, s_Qkp1, &d_G_dense_i[(states_sq+controls_sq)*(k+1)]);
            glass::copy<T>(state_size, s_qk, &d_g_i[states_s_controls*k]);
            glass::copy<T>(control_size, s_rk, &d_g_i[states_s_controls*k+state_size]);
            glass::copy<T>(state_size, s_qkp1, &d_g_i[states_s_controls*(k+1)]);
            glass::copy<T>(states_sq, static_cast<T>(-1), s_Ak, &d_C_dense_i[(states_sq+states_p_controls)*k]);
            glass::copy<T>(states_p_controls, static_cast<T>(-1), s_Bk, &d_C_dense_i[(states_sq+states_p_controls)*k+states_sq]);
            glass::copy<T>(state_size, s_integrator_error, &d_c_i[state_size*(k+1)]);

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
 
            glass::copy<T>(states_sq, s_Qk, &d_G_dense_i[(states_sq+controls_sq)*k]);
            glass::copy<T>(controls_sq, s_Rk, &d_G_dense_i[(states_sq+controls_sq)*k+states_sq]);
            glass::copy<T>(state_size, s_qk, &d_g_i[states_s_controls*k]);
            glass::copy<T>(control_size, s_rk, &d_g_i[states_s_controls*k+state_size]);
            glass::copy<T>(states_sq, static_cast<T>(-1), s_Ak, &d_C_dense_i[(states_sq+states_p_controls)*k]);
            glass::copy<T>(states_p_controls, static_cast<T>(-1), s_Bk, &d_C_dense_i[(states_sq+states_p_controls)*k+states_sq]);
            glass::copy<T>(state_size, s_integrator_error, &d_c_i[state_size*(k+1)]);
        }
    }
}
