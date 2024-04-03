
#include "dynamics/rbd_plant.cuh"
#include "merit.cuh"

template <typename T>
size_t get_kkt_smem_size(uint32_t state_size, uint32_t control_size){
    const uint32_t states_sq = state_size * state_size;
    const uint32_t controls_sq = control_size * control_size;

    size_t smem_size = sizeof(T)*(3*states_sq + 
                                  controls_sq + 
                                  7 * state_size + 
                                  3 * control_size + 
                                  state_size*control_size + 
                                  max(grid::EE_POS_SHARED_MEM_COUNT, grid::DEE_POS_SHARED_MEM_COUNT) + 
                                  max((state_size/2)*(state_size + control_size + 1) + gato_plant::forwardDynamicsAndGradient_TempMemSize_Shared(), 3 + (state_size/2)*6));

    return smem_size;
}


template <typename T, unsigned INTEGRATOR_TYPE = 0, bool ANGLE_WRAP = false>
__global__
void generate_kkt_submatrices(uint32_t state_size, 
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
