#pragma once
// // values assumed coming from an instance of grid
// namespace grid{
// 	//
// 	// TODO do I need all of these?
// 	//

// 	const int NUM_JOINTS = 30;
//     const int ID_DYNAMIC_SHARED_MEM_COUNT = 2340;
//     const int MINV_DYNAMIC_SHARED_MEM_COUNT = 9210;
//     const int FD_DYNAMIC_SHARED_MEM_COUNT = 10110;
//     const int ID_DU_DYNAMIC_SHARED_MEM_COUNT = 10980;
//     const int FD_DU_DYNAMIC_SHARED_MEM_COUNT = 10980;
//     const int ID_DU_MAX_SHARED_MEM_COUNT = 13410;
//     const int FD_DU_MAX_SHARED_MEM_COUNT = 16140;
//     const int SUGGESTED_THREADS = 512;

// 	template <typename T>
//     struct robotModel {
//         T *d_XImats;
//         int *d_topology_helpers;
//     };
// }

#include <stdio.h>
#include <cuda.h> 
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>
#include "iiwa_eepos_grid.cuh"
#include "settings.cuh"

#include "glass.cuh"

// #include <random>
// #define RANDOM_MEAN 0
// #define RANDOM_STDEV 0.001
// std::default_random_engine randEng(time(0)); //seed
// std::normal_distribution<double> randDist(RANDOM_MEAN, RANDOM_STDEV); //mean followed by stdiv

namespace gato_plant{


	const unsigned SUGGESTED_THREADS = grid::SUGGESTED_THREADS;

	template<class T>
	__host__ __device__
	constexpr T PI() {return static_cast<T>(3.14159);}
	template<class T>
	__host__ __device__
	constexpr T GRAVITY() {return static_cast<T>(0.0);}


	// template<class T>
	// __host__ __device__
	// constexpr T COST_Q1() {return static_cast<T>(Q_COST);}
	
	template<class T>
	__host__ __device__
	constexpr T COST_QD() {return static_cast<T>(QD_COST);}

	template<class T>
	__host__ __device__
	constexpr T COST_R() {return static_cast<T>(R_COST);}

	template <typename T>
	void *initializeDynamicsConstMem(){
		grid::robotModel<T> *d_robotModel = grid::init_robotModel<T>();
		return (void *)d_robotModel;
	}
	template <typename T>
	void freeDynamicsConstMem(void *d_dynMem_const){
		grid::free_robotModel((grid::robotModel<T>*) d_dynMem_const);
	}

	// Start at q = [0,0,-0.25*PI,0,0.25*PI,0.5*PI,0] with small random for qd, u, lambda
	// template <typename T>
	// __host__
	// void loadInitialState(T *x){
	// 	T q[7] = {PI<T>(),0.25*PI<T>(),0.167*PI<T>(),-0.167*PI<T>(),PI<T>(),0.167*PI<T>(),0.5*PI<T>()};
	// 	for (int i = 0; i < 7; i++){
	// 		x[i] = q[i]; x[i + 7] = 0;
	// 	}
	// }

	// template <typename T>
	// __host__
	// void loadInitialControl(T *u){for (int i = 0; i < 7; i++){u[i] = 0;}}

	// // goal at q = [-0.5*PI,0.25*PI,0.167*PI,-0.167*PI,0.125*PI,0.167*PI,0.5*PI] with 0 for qd, u, lambda
	// template <typename T>
	// __host__
	// void loadGoalState(T *xg){
	// 	T q[7] = {0,0,-0.25*PI<T>(),0,0.25*PI<T>(),0.5*PI<T>(),0};
	// 	for (int i = 0; i < 7; i++){
	// 		xg[i] = q[i]; xg[i + 7] = static_cast<T>(0);
	// 	}
	// }

	template <typename T>
	__device__
	void forwardDynamics(T *s_qdd, T *s_q, T *s_qd, T *s_u, T *s_XITemp, void *d_dynMem_const, cooperative_groups::thread_block block){

		T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[1008];
    	grid::load_update_XImats_helpers<T>(s_XImats, s_q, (grid::robotModel<float> *) d_dynMem_const, s_temp);
    	__syncthreads();

    	grid::forward_dynamics_inner<T>(s_qdd, s_q, s_qd, s_u, s_XImats, s_temp, gato_plant::GRAVITY<T>());
		
		// grid::forward_dynamics_device<T>(s_qdd,s_q,s_qd,s_u,(grid::robotModel<T>*)d_dynMem_const,GRAVITY<T>());
	}

	__host__ __device__
	constexpr unsigned forwardDynamics_TempMemSize_Shared(){return grid::FD_DYNAMIC_SHARED_MEM_COUNT;}

	// template <typename T>
	// __device__
	// void forwardDynamicsGradient( T *s_dqdd, T *s_q, T *s_qd, T *s_u, T *s_temp, void *d_dynMem_const, cooperative_groups::thread_block block){
	// 	grid::forward_dynamics_gradient_device<T,true>(s_dqdd, s_q, s_qd, s_u, s_temp, (grid::robotModel<T> *)d_dynMem_const,GRAVITY<T>());
	// }

	// __host__ __device__
	// constexpr unsigned forwardDynamicsGradient_TempMemSize_Shared(){return grid::FD_DU_MAX_SHARED_MEM_COUNT;}


    template <typename T, bool INCLUDE_DU = true>
    __device__
    void forwardDynamicsAndGradient(T *s_df_du, T *s_qdd, const T *s_q, const T *s_qd, const T *s_u, T *s_temp_in, void *d_dynMem_const){

		T *s_XITemp = s_temp_in;
		grid::robotModel<T> *d_robotModel = (grid::robotModel<T> *) d_dynMem_const;

        T *s_XImats = s_XITemp; T *s_vaf = &s_XITemp[504]; T *s_dc_du = &s_vaf[126]; T *s_Minv = &s_dc_du[98]; T *s_temp = &s_Minv[49];
        grid::load_update_XImats_helpers<T>(s_XImats, s_q, d_robotModel, s_temp); __syncthreads();
        //TODO: there is a slightly faster way as s_v does not change -- thus no recompute needed
        grid::direct_minv_inner<T>(s_Minv, s_q, s_XImats, s_temp); __syncthreads();
        T *s_c = s_temp;
        grid::inverse_dynamics_inner<T>(s_c, s_vaf, s_q, s_qd, s_XImats, &s_temp[7], GRAVITY<T>()); __syncthreads();
        grid::forward_dynamics_finish<T>(s_qdd, s_u, s_c, s_Minv); __syncthreads();
        grid::inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_qdd, s_XImats, s_temp, GRAVITY<T>()); __syncthreads();
        grid::inverse_dynamics_gradient_inner<T>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_temp, GRAVITY<T>()); __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
            int row = ind % 7; int dc_col_offset = ind - row;
            // account for the fact that Minv is an SYMMETRIC_UPPER triangular matrix
            T val = static_cast<T>(0);
            for(int col = 0; col < 7; col++) {
                int index = (row <= col) * (col * 7 + row) + (row > col) * (row * 7 + col);
                val += s_Minv[index] * s_dc_du[dc_col_offset + col];
            }
            s_df_du[ind] = -val;
            if (INCLUDE_DU && ind < 49){
                int col = ind / 7; int index = (row <= col) * (col * 7 + row) + (row > col) * (row * 7 + col);
                s_df_du[ind + 98] = s_Minv[index];
            }
        }
    }


	// template <typename T>
	// __device__
    // void forwardDynamicsAndGradient(T *s_dqdd, T *s_qdd, T *s_q, T *s_qd, T *s_u,  T *s_temp_in, void *d_dynMem_const, cooperative_groups::thread_block block){
       
		// grid::robotModel<T> *d_robotModel = (grid::robotModel<T> *) d_dynMem_const;
		
		// T *s_dc_du = s_temp_in;
		// T *s_vaf = s_dc_du + 392;
		// T *s_Minv = s_vaf + 252;
		// T *s_XITemp = s_Minv + 196;
		// T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[1008];


	    // grid::load_update_XImats_helpers<T>(s_XImats, s_q, d_robotModel, s_temp);
		
		// grid::direct_minv_inner<T>(s_Minv, s_q, s_XImats, s_temp);
		// grid::inverse_dynamics_inner<T>(s_temp, s_vaf, s_q, s_qd, s_XImats, &s_temp[14], GRAVITY<T>());
		// grid::forward_dynamics_finish<T>(s_qdd, s_u, s_temp, s_Minv);
		
		// grid::inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_qdd, s_XImats, s_temp, GRAVITY<T>());
		// grid::inverse_dynamics_gradient_inner<T>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_temp, GRAVITY<T>());
		// for(int ind = threadIdx.x; ind < 392; ind += blockDim.x){
		// 	int row = ind % 14; int dc_col_offset = ind - row;
		// 	// account for the fact that Minv is an SYMMETRIC_UPPER triangular matrix
		// 	T val = static_cast<T>(0);
		// 	for(int col = 0; col < 14; col++) {
		// 		int index = (row <= col) * (col * 14 + row) + (row > col) * (row * 14 + col);
		// 		val += s_Minv[index] * s_dc_du[dc_col_offset + col];
		// 	}
		// 	s_temp[ind] = -val;
		// }

		// for(int ind = threadIdx.x; ind < 392; ind += blockDim.x){
		// 	s_dqdd[ind] = s_temp[ind];
		// }
		// __syncthreads();
		

		// T *s_XITemp = s_temp_in;
		// grid::robotModel<T> *d_robotModel = (grid::robotModel<T> *) d_dynMem_const;
		// T *s_XImats = s_XITemp; T *s_vaf = &s_XITemp[504]; T *s_dc_du = &s_vaf[126]; T *s_Minv = &s_dc_du[98]; T *s_temp = &s_Minv[49];
        // grid::load_update_XImats_helpers<T>(s_XImats, s_q, d_robotModel, s_temp); __syncthreads();
        // //TODO: there is a slightly faster way as s_v does not change -- thus no recompute needed
        // grid::direct_minv_inner<T>(s_Minv, s_q, s_XImats, s_temp); __syncthreads();
        // T *s_c = s_temp;
        // grid::inverse_dynamics_inner<T>(s_c, s_vaf, s_q, s_qd, s_XImats, &s_temp[7], GRAVITY<T>()); __syncthreads();
        // grid::forward_dynamics_finish<T>(s_qdd, s_u, s_c, s_Minv); __syncthreads();
        // grid::inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_qdd, s_XImats, s_temp, GRAVITY<T>()); __syncthreads();
        // grid::inverse_dynamics_gradient_inner<T>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_temp, GRAVITY<T>()); __syncthreads();
        // for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
        //     int row = ind % 7; int dc_col_offset = ind - row;
        //     // account for the fact that Minv is an SYMMETRIC_UPPER triangular matrix
        //     T val = static_cast<T>(0);
        //     for(int col = 0; col < 7; col++) {
        //         int index = (row <= col) * (col * 7 + row) + (row > col) * (row * 7 + col);
        //         val += s_Minv[index] * s_dc_du[dc_col_offset + col];
        //     }
        //     s_dqdd[ind] = -val;
        //     if (1 && ind < 49){
        //         int col = ind / 7; int index = (row <= col) * (col * 7 + row) + (row > col) * (row * 7 + col);
        //         s_dqdd[ind + 98] = s_Minv[index];
        //     }
        // }



		// grid::robotModel<T> *d_robotModel = (grid::robotModel<T> *) d_dynMem_const;
		// grid::forward_dynamics_gradient_device<T>(s_dqdd, s_q, s_qd, s_u, d_robotModel, GRAVITY<T>());
    // }


	__host__ __device__
	constexpr unsigned forwardDynamicsAndGradient_TempMemSize_Shared(){return grid::FD_DU_MAX_SHARED_MEM_COUNT;}


	__host__
	unsigned trackingcost_TempMemCt_Shared(uint32_t state_size, uint32_t control_size, uint32_t knot_points){
		return state_size/2 + control_size + 3 + 6 + grid::EE_POS_SHARED_MEM_COUNT;
	}

	///TODO: get rid of divergence
		template <typename T>
	__device__
	T trackingcost(uint32_t state_size, uint32_t control_size, uint32_t knot_points, T *s_xu, T *s_eePos_traj, T *s_temp, const grid::robotModel<T> *d_robotModel){
		
        // const T Q_cost = COST_Q1<T>();
		const T QD_cost = COST_QD<T>();
		const T R_cost = COST_R<T>();
        
        T err;
        T val = 0;
		
        // QD and R penalty
		const uint32_t threadsNeeded = state_size/2 + control_size * (blockIdx.x < knot_points - 1);
        
		T *s_cost_vec = s_temp;
		T *s_eePos_cost = s_cost_vec + threadsNeeded + 3;
        T *s_extra_temp = s_eePos_cost + 6;




        for(int i = threadIdx.x; i < threadsNeeded; i += blockDim.x){
			if(i < state_size/2){
                err = s_xu[i + state_size/2];
                val = QD_cost * err * err;
			}
			else{
				err = s_xu[i+state_size/2];
				val = R_cost * err * err;
			}
			s_cost_vec[i] = static_cast<T>(0.5) * val;
		}

        __syncthreads();
        grid::end_effector_positions_device<T>(s_eePos_cost, s_xu, s_extra_temp, d_robotModel);
        __syncthreads();
        
		// if(threadIdx.x==0){
		// 	printf("block %d with input %f,%f,%f,%f,%f,%f,%f\n", blockIdx.x, s_xu[7],s_xu[8],s_xu[9],s_xu[10],s_xu[11],s_xu[12],s_xu[13]);
		// }

        for(int i = threadIdx.x; i < 3; i+=blockDim.x){
            err = s_eePos_cost[i] - s_eePos_traj[i];
            s_cost_vec[threadsNeeded + i] = static_cast<T>(0.5) * err * err;
        }
		__syncthreads();
		glass::reduce<T>(3 + threadsNeeded, s_cost_vec);
		__syncthreads();
		
        return s_cost_vec[0];
	}	


	///TODO: costgradientandhessian could be much faster with no divergence
	// not last block
	template <typename T, bool computeR=true>
	__device__
	void trackingCostGradientAndHessian(uint32_t state_size, 
										uint32_t control_size, 
										T *s_xu, 
										T *s_eePos_traj, 
										T *s_Qk, 
										T *s_qk, 
										T *s_Rk, 
										T *s_rk,
										T *s_temp,
										void *d_robotModel)
	{	
		// const T Q_cost = COST_Q1<T>();
		const T QD_cost = COST_QD<T>();
		const T R_cost = COST_R<T>();

		T *s_eePos = s_temp;
		T *s_eePos_grad = s_eePos + 6;
		T *s_scratch = s_eePos_grad + 6 * state_size/2;

		const uint32_t threads_needed = state_size + control_size*computeR;
		uint32_t offset;
		T x_err, y_err, z_err, err;

		grid::end_effector_positions_device<T>(s_eePos, s_xu, s_scratch, (grid::robotModel<T> *)d_robotModel);
        __syncthreads();
		grid::end_effector_positions_gradient_device<T>(s_eePos_grad, s_xu, s_scratch, (grid::robotModel<T> *)d_robotModel);
        __syncthreads();

		// if(threadIdx.x==0){
		// 	printf("block %d with input %f,%f,%f,%f,%f,%f,%f\n", blockIdx.x, s_xu[0],s_xu[1],s_xu[2],s_xu[3],s_xu[4],s_xu[5],s_xu[6]);
		// }

		for (int i = threadIdx.x; i < threads_needed; i += blockDim.x){
			
			if(i < state_size){
				//gradient
				if (i < state_size / 2){
					// sum x, y, z error
					x_err = (s_eePos[0] - s_eePos_traj[0]);
					y_err = (s_eePos[1] - s_eePos_traj[1]);
					z_err = (s_eePos[2] - s_eePos_traj[2]);

					s_qk[i] = s_eePos_grad[6 * i + 0] * x_err + s_eePos_grad[6 * i + 1] * y_err + s_eePos_grad[6 * i + 2] * z_err;
				}
				else{
					err = s_xu[i];
					s_qk[i] = QD_cost * err;
				}
				
			}
			else{
				err = s_xu[i];
				offset = i - state_size;
				
				//gradient
				s_rk[offset] = R_cost * err;
			}
		}

		__syncthreads();

		for (int i = threadIdx.x; i < threads_needed; i += blockDim.x){
			if (i < state_size){
				//hessian
				for(int j = 0; j < state_size; j++){
					if(j < state_size / 2 && i < state_size / 2){
						s_Qk[i*state_size + j] = s_qk[i] * s_qk[j];
					}
					else{
						s_Qk[i*state_size + j] = (i == j) ? QD_cost : static_cast<T>(0);
					}
				}
			}
			else{
				offset = i - state_size;
				//hessian
				for(int j = 0; j < control_size; j++){
					s_Rk[offset*control_size+j] = (offset == j) ? R_cost : static_cast<T>(0);
				}
			}
		}
	}

	// last block
	template <typename T>
	__device__
	void trackingCostGradientAndHessian_lastblock(uint32_t state_size, 
							    				  uint32_t control_size, 
							    				  T *s_xux, 
							    				  T *s_eePos_traj, 
							    				  T *s_Qk, 
							    				  T *s_qk, 
							    				  T *s_Rk, 
							    				  T *s_rk, 
							    				  T *s_Qkp1, 
							    				  T *s_qkp1,
							    				  T *s_temp,
												  void *d_dynMem_const
												  )
	{
		trackingCostGradientAndHessian<T>(state_size, control_size, s_xux, s_eePos_traj, s_Qk, s_qk, s_Rk, s_rk, s_temp, d_dynMem_const);
		__syncthreads();
		trackingCostGradientAndHessian<T, false>(state_size, control_size, s_xux, &s_eePos_traj[6], s_Qkp1, s_qkp1, nullptr, nullptr, s_temp, d_dynMem_const);
		__syncthreads();
	}

	// __host__ __device__
	// constexpr unsigned costGradientAndHessian_TempMemSize_Shared(){return 0;}
}

