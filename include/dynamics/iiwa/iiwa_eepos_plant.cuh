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
#include "grid.cuh"
#include "settings.cuh"

#include "glass.cuh"

// #include <random>
// #define RANDOM_MEAN 0
// #define RANDOM_STDEV 0.001
// std::default_random_engine randEng(time(0)); //seed
// std::normal_distribution<double> randDist(RANDOM_MEAN, RANDOM_STDEV); //mean followed by stdiv

namespace gato_plant{


	const unsigned SUGGESTED_THREADS = grid::MAX_PERF_LEVEL_THREADS;   // was grid::SUGGESTED_THREADS (renamed in GRiD)

	template<class T>
	__host__ __device__
	constexpr T PI() {return static_cast<T>(3.14159);}
	template<class T>
	__host__ __device__
	// Physical gravity, matching GATO and the pinocchio sim convention (GRiD applies +g as an
	// upward base accel, so -9.81 = physical downward gravity). The solver and the sim both use
	// this value, and the shared reference trajectory is generated with it, so all are consistent.
	constexpr T GRAVITY() {return static_cast<T>(-9.81);}

	// Dimension aliases for the grid_plant::tracking_cost adapters (mirror GATO's plant).
	inline constexpr int NQ  = grid::NUM_JOINTS;      // joints
	inline constexpr int NX  = 2 * grid::NUM_JOINTS;  // state = [q; qd]
	inline constexpr int NU  = grid::NUM_JOINTS;      // controls
	inline constexpr int NEE = grid::NUM_EES;         // end-effector count

	template<class T>
	__host__ __device__
	constexpr T JOINT_LIMIT_MARGIN() {return static_cast<T>(-0.1);}

	// iiwa14 joint/velocity/control limits (from iiwa14.urdf), used by the barrier terms of
	// grid_plant::tracking_cost. A negative margin tightens the usable range inside the hard limit.
	template<class T>
	__device__ constexpr T JOINT_LIMITS_DATA[7][2] = {
	    {-2.96706 - JOINT_LIMIT_MARGIN<T>(), 2.96706 + JOINT_LIMIT_MARGIN<T>()},
	    {-2.09440 - JOINT_LIMIT_MARGIN<T>(), 2.09440 + JOINT_LIMIT_MARGIN<T>()},
	    {-2.96706 - JOINT_LIMIT_MARGIN<T>(), 2.96706 + JOINT_LIMIT_MARGIN<T>()},
	    {-2.09440 - JOINT_LIMIT_MARGIN<T>(), 2.09440 + JOINT_LIMIT_MARGIN<T>()},
	    {-2.96706 - JOINT_LIMIT_MARGIN<T>(), 2.96706 + JOINT_LIMIT_MARGIN<T>()},
	    {-2.09440 - JOINT_LIMIT_MARGIN<T>(), 2.09440 + JOINT_LIMIT_MARGIN<T>()},
	    {-3.05433 - JOINT_LIMIT_MARGIN<T>(), 3.05433 + JOINT_LIMIT_MARGIN<T>()}
	};
	template<class T>
	__device__ constexpr T VEL_LIMITS_DATA[7][2] = {
	    {-1.48353 - JOINT_LIMIT_MARGIN<T>(), 1.48353 + JOINT_LIMIT_MARGIN<T>()},
	    {-1.48353 - JOINT_LIMIT_MARGIN<T>(), 1.48353 + JOINT_LIMIT_MARGIN<T>()},
	    {-1.74533 - JOINT_LIMIT_MARGIN<T>(), 1.74533 + JOINT_LIMIT_MARGIN<T>()},
	    {-1.30900 - JOINT_LIMIT_MARGIN<T>(), 1.30900 + JOINT_LIMIT_MARGIN<T>()},
	    {-2.26893 - JOINT_LIMIT_MARGIN<T>(), 2.26893 + JOINT_LIMIT_MARGIN<T>()},
	    {-2.35619 - JOINT_LIMIT_MARGIN<T>(), 2.35619 + JOINT_LIMIT_MARGIN<T>()},
	    {-2.35619 - JOINT_LIMIT_MARGIN<T>(), 2.35619 + JOINT_LIMIT_MARGIN<T>()}
	};
	template<class T>
	__device__ constexpr T CTRL_LIMITS_DATA[7][2] = {
	    {-320.0 - JOINT_LIMIT_MARGIN<T>(), 320.0 + JOINT_LIMIT_MARGIN<T>()},
	    {-320.0 - JOINT_LIMIT_MARGIN<T>(), 320.0 + JOINT_LIMIT_MARGIN<T>()},
	    {-176.0 - JOINT_LIMIT_MARGIN<T>(), 176.0 + JOINT_LIMIT_MARGIN<T>()},
	    {-176.0 - JOINT_LIMIT_MARGIN<T>(), 176.0 + JOINT_LIMIT_MARGIN<T>()},
	    {-110.0 - JOINT_LIMIT_MARGIN<T>(), 110.0 + JOINT_LIMIT_MARGIN<T>()},
	    { -40.0 - JOINT_LIMIT_MARGIN<T>(),  40.0 + JOINT_LIMIT_MARGIN<T>()},
	    { -40.0 - JOINT_LIMIT_MARGIN<T>(),  40.0 + JOINT_LIMIT_MARGIN<T>()}
	};
	template<class T> __host__ __device__ constexpr const T (&JOINT_LIMITS())[7][2] { return JOINT_LIMITS_DATA<T>; }
	template<class T> __host__ __device__ constexpr const T (&VEL_LIMITS())[7][2]   { return VEL_LIMITS_DATA<T>; }
	template<class T> __host__ __device__ constexpr const T (&CTRL_LIMITS())[7][2]  { return CTRL_LIMITS_DATA<T>; }

	// Joint-posture target q_nom for the Q_COST joint-position term = the reference start config q_start
	// (the generator's q0). With Q_COST>0 this anchors the EE-nullspace joints (esp. joint 7) and makes
	// the state Hessian full-rank PD so the cooperative PCG stays well-posed. (For a moving joint-SPACE
	// tracking cost, s_x_des is instead fed the per-knot reference — see the d_xs_goal path.)
	template<class T>
	__device__ constexpr T Q_NOM_DATA[7] = {0.40, 0.80, 0.30, -1.10, 0.50, 0.60, 0.30};
	template<class T> __host__ __device__ constexpr const T (&Q_NOM())[7] { return Q_NOM_DATA<T>; }

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

		// TOPOLOGY_HELPERS_COUNT == 0 for iiwa14 (fixed serial chain); d_workspace/d_f_ext null,
		// gravity is now the trailing arg. XImats arena is 504 (matches GATO's regenerated grid.cuh).
		int *s_topology_helpers = nullptr;
		T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
    	grid::load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, (grid::robotModel<T> *) d_dynMem_const, s_temp);
    	__syncthreads();

    	grid::forward_dynamics_inner<T>(s_qdd, s_q, s_qd, s_u, s_XImats, s_topology_helpers, s_temp, /*d_workspace*/nullptr, /*d_f_ext*/nullptr, gato_plant::GRAVITY<T>());
	}

	__host__ __device__
	constexpr unsigned forwardDynamics_TempMemSize_Shared(){return grid::FORWARD_DYNAMICS_DYNAMIC_SHARED_MEM_COUNT;}

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

        int *s_topology_helpers = nullptr;   // TOPOLOGY_HELPERS_COUNT == 0 (iiwa14 fixed chain)
        T *s_XImats = s_XITemp; T *s_vaf = &s_XITemp[504]; T *s_dc_du = &s_vaf[126]; T *s_Minv = &s_dc_du[98]; T *s_temp = &s_Minv[49];
        grid::load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp); __syncthreads();
        //TODO: there is a slightly faster way as s_v does not change -- thus no recompute needed
        grid::minv_inner<T>(s_Minv, s_q, s_XImats, s_topology_helpers, s_temp, /*d_workspace*/nullptr); __syncthreads();
        T *s_c = s_temp;
        grid::inverse_dynamics_inner<T>(s_c, s_vaf, s_q, s_qd, s_XImats, s_topology_helpers, &s_temp[7], /*d_f_ext*/nullptr, GRAVITY<T>()); __syncthreads();
        grid::forward_dynamics_finish<T>(s_qdd, s_u, s_c, s_Minv); __syncthreads();
        grid::inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_qdd, s_XImats, s_topology_helpers, s_temp, /*d_f_ext*/nullptr, GRAVITY<T>()); __syncthreads();
        grid::inverse_dynamics_gradient_inner<T>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_topology_helpers, s_temp, /*d_temp_spill*/nullptr, GRAVITY<T>()); __syncthreads();
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


	// ===================================================================================
	// grid_plant::tracking_cost ADAPTER (ported from GATO's plant; same unified cost recipe:
	// EE-position + quadratic state/input regularization + joint/velocity/torque barriers).
	// Bridges the scalar cost weights + constexpr limits to grid_plant's per-element buffer
	// contract; all buffers are carved from a caller scratch slab and built cooperatively.
	// ===================================================================================

	template<typename T>
	__host__ __device__ constexpr unsigned trackingCostValue_TempMemCt(){
		return (2*NX + 2*NU + 6) + (4*NQ + 2*NU) + 6*NEE
		       + grid::END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_COUNT;
	}
	template<typename T>
	__host__ __device__ constexpr unsigned trackingCostGradHess_TempMemCt(){
		return (2*NX + 2*NU + 6) + (4*NQ + 2*NU) + 6*NEE + 6*NQ*NEE
		       + grid::END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_COUNT;
	}

	template<typename T>
	__device__ void buildTrackingCostBuffers(
	    T* s_Q, T* s_R, T* s_W, T* s_x_des, T* s_u_des, T* s_ee_des,
	    T* s_q_lo, T* s_q_hi, T* s_qd_lo, T* s_qd_hi, T* s_u_lo, T* s_u_hi,
	    const T* s_eePos_traj, T q_state_cost, T qd_cost, T u_cost, T ee_weight)
	{
		const int tid = threadIdx.x + threadIdx.y * blockDim.x;
		const int nth = blockDim.x * blockDim.y;
		// s_Q: q_state_cost on the q-block (joint-posture toward Q_NOM; 0 => EE-only), qd_cost on the
		// qd-block. A nonzero q_state_cost makes the state Hessian full-rank PD (PCG-robust).
		for (int i = tid; i < NX; i += nth) {
			if (i < NQ) { s_Q[i] = q_state_cost; s_x_des[i] = Q_NOM<T>()[i]; }
			else        { s_Q[i] = qd_cost;      s_x_des[i] = static_cast<T>(0); }
		}
		for (int i = tid; i < NU; i += nth) { s_R[i] = u_cost; s_u_des[i] = static_cast<T>(0); }
		for (int i = tid; i < 3;  i += nth) { s_W[i] = ee_weight; s_ee_des[i] = s_eePos_traj[i]; }
		for (int i = tid; i < NQ; i += nth) {
			s_q_lo[i]  = JOINT_LIMITS<T>()[i][0]; s_q_hi[i]  = JOINT_LIMITS<T>()[i][1];
			s_qd_lo[i] = VEL_LIMITS<T>()[i][0];   s_qd_hi[i] = VEL_LIMITS<T>()[i][1];
		}
		for (int i = tid; i < NU; i += nth) { s_u_lo[i] = CTRL_LIMITS<T>()[i][0]; s_u_hi[i] = CTRL_LIMITS<T>()[i][1]; }
	}

	// VALUE adapter. is_terminal picks N_cost EE weight + drops control reg/barrier.
	template<typename T>
	__device__ T trackingCostValue(
	    const T* s_x, const T* s_u, const T* s_eePos_traj, T* s_temp,
	    const grid::robotModel<T>* d_robotModel,
	    T q_cost, T q_state_cost, T qd_cost, T u_cost, T N_cost,
	    T q_lim_cost, T vel_lim_cost, T ctrl_lim_cost, bool is_terminal)
	{
		T* s_Q = s_temp;          T* s_R = s_Q + NX;          T* s_W = s_R + NU;
		T* s_x_des = s_W + 3;     T* s_u_des = s_x_des + NX;  T* s_ee_des = s_u_des + NU;
		T* s_q_lo = s_ee_des + 3; T* s_q_hi = s_q_lo + NQ;
		T* s_qd_lo = s_q_hi + NQ; T* s_qd_hi = s_qd_lo + NQ;
		T* s_u_lo = s_qd_hi + NQ; T* s_u_hi = s_u_lo + NU;
		T* s_eePos = s_u_hi + NU; T* s_scratch = s_eePos + 6 * NEE;

		const T ee_w = is_terminal ? N_cost : q_cost;
		const T u_w  = is_terminal ? static_cast<T>(0) : u_cost;
		const T mu_u = is_terminal ? static_cast<T>(0) : ctrl_lim_cost;
		buildTrackingCostBuffers<T>(s_Q, s_R, s_W, s_x_des, s_u_des, s_ee_des,
		                            s_q_lo, s_q_hi, s_qd_lo, s_qd_hi, s_u_lo, s_u_hi,
		                            s_eePos_traj, q_state_cost, qd_cost, u_w, ee_w);
		__syncthreads();
		__shared__ T s_out[1];
		grid_plant::tracking_cost<T, 0>(s_out, s_x, s_u, s_x_des, s_u_des, s_ee_des,
		                                s_Q, s_R, s_W, s_q_lo, s_q_hi, q_lim_cost,
		                                s_qd_lo, s_qd_hi, vel_lim_cost, s_u_lo, s_u_hi, mu_u,
		                                s_eePos, s_scratch, d_robotModel);
		__syncthreads();
		return s_out[0];
	}

	// GRAD+HESS adapter. ee_weight = q_cost (running, at s_x) or N_cost (terminal, at x_{k+1}).
	// For a terminal R-less call, pass throwaway s_rk/s_Rk.
	template<typename T>
	__device__ void trackingCostGradHess(
	    const T* s_x, const T* s_u, const T* s_eePos_traj,
	    T* s_Qk, T* s_qk, T* s_Rk, T* s_rk, T* s_temp,
	    const grid::robotModel<T>* d_robotModel,
	    T q_state_cost, T qd_cost, T u_cost, T q_lim_cost, T vel_lim_cost, T ctrl_lim_cost, T ee_weight)
	{
		T* s_Q = s_temp;          T* s_R = s_Q + NX;          T* s_W = s_R + NU;
		T* s_x_des = s_W + 3;     T* s_u_des = s_x_des + NX;  T* s_ee_des = s_u_des + NU;
		T* s_q_lo = s_ee_des + 3; T* s_q_hi = s_q_lo + NQ;
		T* s_qd_lo = s_q_hi + NQ; T* s_qd_hi = s_qd_lo + NQ;
		T* s_u_lo = s_qd_hi + NQ; T* s_u_hi = s_u_lo + NU;
		T* s_eePos = s_u_hi + NU; T* s_eePosGrad = s_eePos + 6 * NEE;
		T* s_scratch = s_eePosGrad + 6 * NQ * NEE;

		buildTrackingCostBuffers<T>(s_Q, s_R, s_W, s_x_des, s_u_des, s_ee_des,
		                            s_q_lo, s_q_hi, s_qd_lo, s_qd_hi, s_u_lo, s_u_hi,
		                            s_eePos_traj, q_state_cost, qd_cost, u_cost, ee_weight);
		__syncthreads();
		// grid_plant writes s_Qk/s_Rk column-major; the tracking Hessian is symmetric so the
		// row-major consumers see an identical matrix.
		grid_plant::tracking_cost_gradient<T, 0>(s_qk, s_rk, s_x, s_u, s_x_des, s_u_des, s_ee_des,
		                                         s_Q, s_R, s_W, s_q_lo, s_q_hi, q_lim_cost,
		                                         s_qd_lo, s_qd_hi, vel_lim_cost, s_u_lo, s_u_hi, ctrl_lim_cost,
		                                         s_eePos, s_eePosGrad, s_scratch, d_robotModel);
		__syncthreads();
		grid_plant::tracking_cost_hessian<T, 0>(s_Qk, s_Rk, s_x, s_u,
		                                        s_Q, s_R, s_W, s_q_lo, s_q_hi, q_lim_cost,
		                                        s_qd_lo, s_qd_hi, vel_lim_cost, s_u_lo, s_u_hi, ctrl_lim_cost,
		                                        s_eePosGrad, s_scratch, d_robotModel);
		__syncthreads();
	}

	__host__
	unsigned trackingcost_TempMemCt_Shared(uint32_t state_size, uint32_t control_size, uint32_t knot_points){
		return trackingCostValue_TempMemCt<float>();
	}

	// VALUE wrapper (kkt/merit-facing signature). Delegates to the grid_plant::tracking_cost
	// adapter; the terminal knot (blockIdx == knot_points-1) uses N_COST + drops control reg.
	template <typename T>
	__device__
	T trackingcost(uint32_t state_size, uint32_t control_size, uint32_t knot_points, T *s_xu, T *s_eePos_traj, T *s_temp, const grid::robotModel<T> *d_robotModel){
		return trackingCostValue<T>(s_xu, s_xu + state_size, s_eePos_traj, s_temp, d_robotModel,
		                            static_cast<T>(EE_COST), static_cast<T>(Q_COST), static_cast<T>(QD_COST), static_cast<T>(U_COST),
		                            static_cast<T>(N_COST), static_cast<T>(Q_LIM_COST), static_cast<T>(VEL_LIM_COST),
		                            static_cast<T>(CTRL_LIM_COST), /*is_terminal=*/(blockIdx.x == knot_points - 1));
	}


	///TODO: costgradientandhessian could be much faster with no divergence
	// not last block
	// GRAD+HESS wrapper (running knot, kkt-facing signature). Delegates to the grid_plant adapter
	// with the running EE weight EE_COST. (computeR retained for ABI; grid_plant always builds R.)
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
		trackingCostGradHess<T>(s_xu, s_xu + state_size, s_eePos_traj, s_Qk, s_qk, s_Rk, s_rk, s_temp,
		                        (const grid::robotModel<T> *)d_robotModel,
		                        static_cast<T>(Q_COST), static_cast<T>(QD_COST), static_cast<T>(U_COST), static_cast<T>(Q_LIM_COST),
		                        static_cast<T>(VEL_LIM_COST), static_cast<T>(CTRL_LIM_COST), /*ee_weight=*/static_cast<T>(EE_COST));
	}

	// last block: knot k (running, EE weight EE_COST, at x_k) + terminal knot k+1 (EE weight
	// N_COST, at x_{k+1} = &s_xux[state_size+control_size}). The terminal R block is discarded
	// (no control at the terminal state) — carve throwaway R/r from the head of s_temp.
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
		const grid::robotModel<T> *d_robotModel = (const grid::robotModel<T> *)d_dynMem_const;
		// running knot k
		trackingCostGradHess<T>(s_xux, s_xux + state_size, s_eePos_traj, s_Qk, s_qk, s_Rk, s_rk, s_temp,
		                        d_robotModel,
		                        static_cast<T>(Q_COST), static_cast<T>(QD_COST), static_cast<T>(U_COST), static_cast<T>(Q_LIM_COST),
		                        static_cast<T>(VEL_LIM_COST), static_cast<T>(CTRL_LIM_COST), /*ee_weight=*/static_cast<T>(EE_COST));
		__syncthreads();
		// terminal knot k+1 at x_{k+1} with N_COST; throwaway R/r at the head of s_temp.
		T *s_R_dummy = s_temp;
		T *s_r_dummy = s_R_dummy + control_size * control_size;
		T *s_temp2   = s_r_dummy + control_size;
		T *s_xkp1    = s_xux + state_size + control_size;
		trackingCostGradHess<T>(s_xkp1, s_xkp1, &s_eePos_traj[6], s_Qkp1, s_qkp1, s_R_dummy, s_r_dummy, s_temp2,
		                        d_robotModel,
		                        static_cast<T>(Q_COST), static_cast<T>(QD_COST), static_cast<T>(U_COST), static_cast<T>(Q_LIM_COST),
		                        static_cast<T>(VEL_LIM_COST), static_cast<T>(CTRL_LIM_COST), /*ee_weight=*/static_cast<T>(N_COST));
		__syncthreads();
	}

	// __host__ __device__
	// constexpr unsigned costGradientAndHessian_TempMemSize_Shared(){return 0;}
}

