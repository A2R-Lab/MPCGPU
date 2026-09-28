/**
 * This instance of grid.cuh is optimized for the urdf: KUKAiiwa14
 *
 * Notes:
 *   Interface is:
 *       __host__   robotModel<T> *d_robotModel = init_robotModel<T>()
 *       __host__   cudaStream_t streams = init_grid<T>()
 *       __host__   gridData<T> *hd_ata = init_gridData<T,NUM_TIMESTEPS>();    __host__   close_grid<T>(cudaStream_t *streams, robotModel<T> *d_robotModel, gridData<T> *hd_data)
 *   
 *       __device__ inverse_dynamics_inner<T>(T *s_c,  T *s_vaf, const T *s_q, const T *s_qd, const T *s_qdd, T *s_XImats, int *s_topology_helpers, T *s_temp, const T gravity)
 *       __device__ inverse_dynamics_inner<T>(T *s_c,  T *s_vaf, const T *s_q, const T *s_qd, T *s_XImats, int *s_topology_helpers, T *s_temp, const T gravity)
 *       __device__ inverse_dynamics_device<T>(T *s_c, const T *s_q, const T *s_qd, const robotModel<T> *d_robotModel, const T gravity)
 *       __device__ inverse_dynamics_device<T>(T *s_c, const T *s_q, const T *s_qd, const T *s_qdd, const robotModel<T> *d_robotModel, const T gravity)
 *       __global__ inverse_dynamics_kernel<T>(T *d_c, const T *d_q_qd, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS)
 *       __global__ inverse_dynamics_kernel<T>(T *d_c, const T *d_q_qd, const T *d_qdd, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS)
 *       __host__   inverse_dynamics<T,USE_QDD_FLAG=false,USE_COMPRESSED_MEM=false>(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps, const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams)
 *   
 *       __device__ inverse_dynamics_inner_vaf<T>(T *s_vaf, const T *s_q, const T *s_qd, const T *s_qdd, T *s_XImats, int *s_topology_helpers, T *s_temp, const T gravity)
 *       __device__ inverse_dynamics_inner_vaf<T>(T *s_vaf, const T *s_q, const T *s_qd, T *s_XImats, int *s_topology_helpers, T *s_temp, const T gravity)
 *       __device__ inverse_dynamics_vaf_device<T>(T *s_vaf, const T *s_q, const T *s_qd, const robotModel<T> *d_robotModel, const T gravity)
 *       __device__ inverse_dynamics_vaf_device<T>(T *s_vaf, const T *s_q, const T *s_qd, const T *s_qdd, const robotModel<T> *d_robotModel, const T gravity)
 *   
 *       __device__ minv_inner<T>(T *s_Minv, T *s_F, const T *s_q, T *s_XImats, int *s_topology_helpers, T *s_temp)
 *       __device__ minv_device<T>(T *s_Minv, const T *s_q, const robotModel<T> *d_robotModel)
 *       __global__ minv_Kernel<T>(T *d_Minv, unsigned char *d_workspace, const T *d_q, const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS)
 *       __host__   minv<T,USE_COMPRESSED_MEM=false>(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps, const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams)
 *   
 *       __device__ forward_dynamics_inner<T>(T *s_qdd, const T *s_q, const T *s_qd, const T *s_u, T *s_minv_F, T *s_XImats, int *s_topology_helpers, T *s_temp, const T gravity)
 *       __device__ forward_dynamics_device<T, RESOURCE_TIER=TIER_SHARED>(T *s_qdd, const T *s_q, const T *s_qd, const T *s_u, const robotModel<T> *d_robotModel, const T gravity, T *d_workspace = nullptr)
 *       __global__ forward_dynamics_kernel<T>(T *d_qdd, unsigned char *d_workspace, const T *d_q_qd_u, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS)
 *       __host__   forward_dynamics<T>(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps, const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams)
 *   
 *       __device__ inverse_dynamics_gradient_inner<T>(T *s_dc_du, const T *s_q, const T *s_qd, const T *s_vaf, T *s_XImats, int *s_topology_helpers, T *s_temp, const T gravity)
 *       __device__ inverse_dynamics_gradient_device<T>(T *s_dc_du, const T *s_q, const T *s_qd, const T *robotModel<T> *d_robotModel, const T gravity)
 *       __device__ inverse_dynamics_gradient_device<T>(T *s_dc_du, const T *s_q, const T *s_qd, const T *s_qdd, const robotModel<T> *d_robotModel, const T gravity)
 *       __global__ inverse_dynamics_gradient_kernel<T>(T *d_dc_du, const T *d_q_qd, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS)
 *       __global__ inverse_dynamics_gradient_kernel<T>(T *d_dc_du, const T *d_q_qd, const T *d_qdd, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS)
 *       __host__   inverse_dynamics_gradient<T,USE_QDD_FLAG=false,USE_COMPRESSED_MEM=false>(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps, const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams)
 *   
 *       __device__ forward_dynamics_gradient_device<T>(T *s_df_du, const T *s_q, const T *s_qd, const T *s_u, const robotModel<T> *d_robotModel, const T gravity)
 *       __device__ forward_dynamics_gradient_device<T>(T *s_df_du, const T *s_q, const T *s_qd, const T *s_qdd, const T *s_Minv, const robotModel<T> *d_robotModel, const T gravity)
 *       __global__ forward_dynamics_gradient_kernel<T>(T *d_df_du, const T *d_q_qd_u, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS)
 *       __global__ forward_dynamics_gradient_kernel<T>(T *d_df_du, const T *d_q_qd, const T *d_qdd, const T *d_Minv, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS)
 *       __host__   forward_dynamics_gradient<T,USE_QDD_MINV_FLAG=false>(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps, const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams)
 *   
 *       __device__ end_effector_pose_inner<T,TEMP_IN_SMEM=true>(T *s_end_effector_pose, const T *s_q, const T *s_Xhom, int *s_topology_helpers, T *s_temp, T *d_workspace, unsigned char *s_linalg_smem)
 *       __device__ end_effector_pose_device<T>(T *s_end_effector_pose, const T *s_q, const robotModel<T> *d_robotModel)
 *       __global__ end_effector_pose_kernel<T>(T *d_end_effector_pose, const T *d_q, const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS)
 *       __host__   end_effector_pose<T,USE_COMPRESSED_MEM=false>(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps, const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams)
 *   
 *       __device__ end_effector_pose_gradient_inner<T>(T *s_end_effector_pose_gradient, const T *s_q, const T *s_Xhom, const T *s_dXhom, int *s_topology_helpers, T *s_temp)
 *       __device__ end_effector_pose_gradient_device<T>(T *s_end_effector_pose_gradient, const T *s_q, const robotModel<T> *d_robotModel)
 *       __global__ end_effector_pose_gradient_kernel<T>(T *d_end_effector_pose_gradient, unsigned char *d_workspace, const T *d_q, const int stride_q, const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS)
 *       __host__   end_effector_pose_gradient<T,USE_COMPRESSED_MEM=false>(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps, const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams)
 *   
 *       __device__ end_effector_pose_hessian_inner<T>(T *s_end_effector_pose_gradient, const T *s_q, const T *s_Xhom, const T *s_dXhom, int *s_topology_helpers, T *s_temp)
 *       __device__ end_effector_pose_hessian_device<T>(T *s_end_effector_pose_gradient, const T *s_q, const robotModel<T> *d_robotModel)
 *       __global__ end_effector_pose_hessian_kernel<T>(T *d_end_effector_pose_gradient, const T *d_q, const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS)
 *       __host__   end_effector_pose_hessian<T,USE_COMPRESSED_MEM=false>(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps, const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams)
 *   
 *       __device__ idsva_so_body_frame_inner(T *s_idsva_so, const T *s_q, const T *s_qd, T *s_qdd, T *s_XImats, T *s_mem, const T gravity)
 *       __global__ idsva_so_body_frame_kernel(T *d_idsva_so, const T *d_q_qd_u, const int stride_q_qd_u, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS)
 *       __host__   idsva_so_body_frame<T>(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps, const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams)
 *   
 *       __device__ fdsva_so_contract(T *s_df2, T *s_idsva_so, T *s_Minv, T *s_df_du, T *s_q, T *s_qd, const T *s_qdd, const T *s_tau, T *s_XImats, T *s_temp, const T gravity)
 *       __device__ fdsva_so_device(T *s_df2, T *s_df_du, const T *s_q, const T *s_qd, const T *s_u, const robotModel<T> *d_robotModel, const T gravity)
 *       __global__ fdsva_so_kernel(T *d_df2, const T *d_q_qd_qdd_tau, const int stride_q_qd_qdd, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS)
 *       __host__   fdsva_so<T>(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps, const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams)
 *   
 *   
 *   Suggested Type T is float
 *   
 *   Additional helper functions and ALGORITHM_inner functions which take in __shared__ memory temp variables exist -- see function descriptions in the file
 *   
 *   By default device and kernels need to be launched with dynamic shared mem of size <FUNC_CODE>_DYNAMIC_SHARED_MEM_COUNT where <FUNC_CODE> = [INVERSE_DYNAMICS, MINV, FORWARD_DYNAMICS, INVERSE_DYNAMICS_GRADIENT, FORWARD_DYNAMICS_GRADIENT]
 *   
 *   Codegen profile: all
 *   Generated algorithms: end_effector_pose, end_effector_pose_gradient, forward_dynamics, forward_dynamics_gradient, inverse_dynamics, inverse_dynamics_gradient, minv
 *   
 *   Additional EEPose Functions Included for Fixed Kinematic Target: EE
 *   
 *
 */

#include <assert.h>
#include <cstddef>
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include "glass.cuh"  // vendor_glass=False: the consumer's top-level GLASS (-I<GLASS root>)

#if defined(__has_include)
#if __has_include(<cub/cub.cuh>)
#define GRID_CUB_HEADER_AVAILABLE 1
#else
#define GRID_CUB_HEADER_AVAILABLE 0
#endif
#else
#define GRID_CUB_HEADER_AVAILABLE 0
#endif
// single kernel timing helper code
#define time_delta_us_timespec(start,end) (1e6*static_cast<double>(end.tv_sec - start.tv_sec)+1e-3*static_cast<double>(end.tv_nsec - start.tv_nsec))

#define XIMAT_SIZE 36
/**
 * Check for runtime errors using the CUDA API
 *
 * Notes:
 *   Adapted from https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
 *
 */
__host__ inline cudaError_t* grid_last_error_slot(){ static cudaError_t e = cudaSuccess; return &e; }
__host__ inline cudaError_t grid_last_error(){ return *grid_last_error_slot(); }
__host__ inline cudaError_t grid_consume_last_error(){ cudaError_t e = *grid_last_error_slot(); *grid_last_error_slot() = cudaSuccess; return e; }
__host__
inline void gpuAssert(cudaError_t code, const char *file, const int line, bool abort=true){
    if (code != cudaSuccess){
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        #ifdef GRID_GPUERRCHK_NO_EXIT
        if (abort && *grid_last_error_slot() == cudaSuccess){ *grid_last_error_slot() = code; }
        #else
        if (abort){cudaDeviceReset(); exit(code);}
        #endif
    }
}
#ifndef gpuErrchk
#define gpuErrchk(err) {gpuAssert(err, __FILE__, __LINE__);}
#endif
#ifndef gpuErrchkKernel
#define gpuErrchkKernel() {gpuErrchk(cudaPeekAtLastError()); gpuErrchk(cudaDeviceSynchronize());}
#endif

// ─── library-safe initialization contract (init_*_checked / free_robotModel_checked) ───
// Host-only fault-injection seams: define BEFORE including this header to intercept
// every allocation/copy the checked initializers make (tests); default = the bare call.
#ifndef GRID_CUDA_CALL
#define GRID_CUDA_CALL(expr) (expr)
#endif
#ifndef GRID_HOST_ALLOC
#define GRID_HOST_ALLOC(expr) (expr)
#endif
__host__ inline cudaError_t grid_fail(const char **failed_op, const char *op, cudaError_t code){
    if (failed_op != nullptr && *failed_op == nullptr) { *failed_op = op; }
    return code;
}
__host__ inline void grid_cleanup_free(void *p, const char *op, cudaError_t *first_cleanup_code, const char **first_cleanup_op){
    if (p == nullptr) { return; }
    cudaError_t e = GRID_CUDA_CALL(cudaFree(p));
    if (e != cudaSuccess && first_cleanup_code != nullptr && *first_cleanup_code == cudaSuccess) {
        *first_cleanup_code = e; if (first_cleanup_op != nullptr) { *first_cleanup_op = op; }
    }
}
__host__ inline void grid_legacy_check(cudaError_t e, const char *op, const char *file, const int line){
    if (e != cudaSuccess) { fprintf(stderr, "GRiD: %s failed: ", op ? op : "initialization"); gpuAssert(e, file, line); }
}

template <typename T, int M, int N>
__host__ __device__
void printMat(T *A, int lda){
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){printf("%.4f ",A[i + lda*j]);}
        printf("\n");
    }
}

template <typename T, int M, int N>
__host__ __device__
void printMat(const T *A, int lda){
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){printf("%.4f ",A[i + lda*j]);}
        printf("\n");
    }
}

#define GRID_HAS_IDSVA_SO_BODY_FRAME 0
#define GRID_HAS_FDSVA_SO 0
#define GRID_HAS_IDSVA_SO_WORLD_FRAME 0
#define GRID_HAS_IDSVA_SO 0
#define GRID_IDSVA_SO_DISPATCHES_WORLD_FRAME 0
#define GRID_HAS_INTEGRATOR 0
#define GRID_HAS_INTEGRATOR_GRADIENT 0
#define GRID_HAS_INVERSE_DYNAMICS 1
#define GRID_HAS_MINV 1
#define GRID_HAS_FORWARD_DYNAMICS 1
#define GRID_HAS_ABA 0
#define GRID_HAS_CRBA 0
#define GRID_HAS_INVERSE_DYNAMICS_GRADIENT 1
#define GRID_HAS_FORWARD_DYNAMICS_GRADIENT 1
#define GRID_HAS_F_EXT_GRADIENT 0
#define GRID_HAS_F_EXT_GRADIENT_DQ 0
#define GRID_HAS_INVERSE_DYNAMICS_REGRESSOR 0
#define GRID_HAS_INVERSE_DYNAMICS_REGRESSOR_GRADIENT 0
#define GRID_HAS_FORWARD_DYNAMICS_PARAMETER_GRADIENT 0
#define GRID_HAS_END_EFFECTOR_POSE 1
#define GRID_HAS_END_EFFECTOR_POSE_GRADIENT 1
#define GRID_HAS_END_EFFECTOR_POSE_HESSIAN 0
#define GRID_HAS_GENERALIZED_GRAVITY 0
#define GRID_HAS_NONLINEAR_EFFECTS 0
#define GRID_HAS_CORIOLIS_MATRIX 0
#define GRID_HAS_KINETIC_ENERGY_REGRESSOR 0
#define GRID_HAS_POTENTIAL_ENERGY_REGRESSOR 0

/**
 * All functions are kept in this namespace
 *
 */
namespace grid {
    __host__ __device__ constexpr size_t grid_align_up(size_t offset, size_t alignment) {
        return (offset + alignment - 1) / alignment * alignment;
    }
    
    template <typename U>
    __device__ U *grid_arena_ptr(unsigned char *arena, size_t byte_offset) {
        return reinterpret_cast<U *>(arena + byte_offset);
    }
    
    template <typename T>
    __host__ __device__ constexpr size_t grid_shared_arena_bytes(size_t t_count, size_t int_count = 0, size_t extra_byte_count = 0) {
        size_t offset = 0;
        offset = grid_align_up(offset, alignof(T));
        offset += sizeof(T) * t_count;
        if (int_count > 0) {
            offset = grid_align_up(offset, alignof(int));
            offset += sizeof(int) * int_count;
        }
        if (extra_byte_count > 0) {
            offset = grid_align_up(offset, static_cast<size_t>(16));
            offset += extra_byte_count;
        }
        return grid_align_up(offset, static_cast<size_t>(16));
    }
    
    #ifndef GRID_CUDA_TARGET_SHARED_MEM_BYTES
    #define GRID_CUDA_TARGET_SHARED_MEM_BYTES 98304
    #endif
    
    #ifndef GRID_WORKSPACE_SLOTS
    #define GRID_WORKSPACE_SLOTS 1
    #endif
    
    enum gridDataKind { GRID_DATA_ALL = 0, GRID_DATA_DYNAMICS = 1, GRID_DATA_KINEMATICS = 2 };
    enum gridSharedTier { GRID_SHARED_FULL = 0, GRID_SPILL_DA_DF_OUTPUT = 1, GRID_SPILL_DV_DA_DF_OUTPUT = 2 };
    // Time integrator family selected by integrator kernels at compile time.
    // EULER / SEMI_IMPLICIT_EULER / CONSTANT_ACCELERATION are single-stage; MIDPOINT / TRAPEZOIDAL / RK4
    // are multi-stage (driven inline from integrator_inner). TRAPEZOIDAL = 5 (NOT MIDPOINT=2).
    enum class IntegratorType { EULER = 0, SEMI_IMPLICIT_EULER = 1, MIDPOINT = 2, RK4 = 3, TRAPEZOIDAL = 4, CONSTANT_ACCELERATION = 5 };
    
    #ifndef GRID_CUDA_ENABLE_L2_PERSISTING
    #define GRID_CUDA_ENABLE_L2_PERSISTING 0
    #endif
    
    __host__ inline cudaError_t grid_get_max_dynamic_shared_memory_bytes(size_t *bytes) {
        int device = 0;
        cudaError_t err = cudaGetDevice(&device);
        if (err != cudaSuccess) { return err; }
        int max_per_block = 0;
        err = cudaDeviceGetAttribute(&max_per_block, cudaDevAttrMaxSharedMemoryPerBlock, device);
        if (err != cudaSuccess) { return err; }
        int max_optin = 0;
    #if CUDART_VERSION >= 9000
        err = cudaDeviceGetAttribute(&max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
        if (err != cudaSuccess) { cudaGetLastError(); max_optin = 0; }
    #endif
        *bytes = static_cast<size_t>(max_optin > max_per_block ? max_optin : max_per_block);
        return cudaSuccess;
    }
    
    __host__ inline cudaError_t grid_check_dynamic_shared_memory_bytes(const char *kernel_name, size_t bytes) {
        size_t max_bytes = 0;
        cudaError_t err = grid_get_max_dynamic_shared_memory_bytes(&max_bytes);
        if (err != cudaSuccess) { return err; }
        if (bytes > max_bytes) {
            fprintf(stderr, "GRID shared-memory request for %s is %zu bytes, but this device supports %zu bytes per block\n",
                    kernel_name, bytes, max_bytes);
            return cudaErrorInvalidConfiguration;
        }
        return cudaSuccess;
    }
    
    __host__ inline cudaError_t grid_begin_l2_persisting(cudaStream_t stream, void *ptr, size_t bytes) {
    #if GRID_CUDA_ENABLE_L2_PERSISTING && CUDART_VERSION >= 11000
        if (ptr == nullptr || bytes == 0) { return cudaSuccess; }
        int device = 0;
        cudaError_t err = cudaGetDevice(&device);
        if (err != cudaSuccess) { return err; }
        int max_window = 0;
        err = cudaDeviceGetAttribute(&max_window, cudaDevAttrMaxAccessPolicyWindowSize, device);
        if (err != cudaSuccess || max_window <= 0) { cudaGetLastError(); return cudaSuccess; }
        int max_persisting_l2 = 0;
        err = cudaDeviceGetAttribute(&max_persisting_l2, cudaDevAttrMaxPersistingL2CacheSize, device);
        if (err == cudaSuccess && max_persisting_l2 > 0) {
            size_t l2_bytes = bytes < static_cast<size_t>(max_persisting_l2) ? bytes : static_cast<size_t>(max_persisting_l2);
            cudaError_t limit_err = cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, l2_bytes);
            if (limit_err != cudaSuccess) { cudaGetLastError(); }
        }
        else { cudaGetLastError(); }
        cudaStreamAttrValue attr;
        memset(&attr, 0, sizeof(attr));
        attr.accessPolicyWindow.base_ptr = ptr;
        attr.accessPolicyWindow.num_bytes = bytes < static_cast<size_t>(max_window) ? bytes : static_cast<size_t>(max_window);
        attr.accessPolicyWindow.hitRatio = 0.60;
        attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
        attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
        return cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
    #else
        (void)stream; (void)ptr; (void)bytes;
        return cudaSuccess;
    #endif
    }
    
    __host__ inline cudaError_t grid_end_l2_persisting(cudaStream_t stream) {
    #if GRID_CUDA_ENABLE_L2_PERSISTING && CUDART_VERSION >= 11000
        cudaStreamAttrValue attr;
        memset(&attr, 0, sizeof(attr));
        attr.accessPolicyWindow.num_bytes = 0;
        return cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
    #else
        (void)stream;
        return cudaSuccess;
    #endif
    }
    
    // Workspace slots: at large batch sizes the per-timestep device WORKSPACE (not
    // the outputs) is what overflows device RAM on big robots. init_gridData auto-fits
    // the arena to hd_data->workspace_timestep_slots slots (cudaMemGetInfo; override
    // with the GRID_WORKSPACE_TIMESTEP_SLOTS env var), kernels index the arena by
    // BLOCK slot -- constant per block across its grid-stride timesteps, so a block
    // reuses one slot sequentially and slots never alias across live blocks -- and
    // every workspace-using host wrapper clamps its launch grid to the slot count.
    // Memory-comfortable case: slots == num_timesteps and launches are unchanged.
    __device__ __forceinline__ int grid_workspace_slot() {
        return blockIdx.x + blockIdx.y*gridDim.x;
    }
    // Clamp a requested launch thread count against the LAUNCHED kernel's own
    // cudaFuncAttributes cap (register pressure / __launch_bounds__). A request
    // above the cap is otherwise silently rejected at launch time: the stream
    // stays empty, sync succeeds, and the output buffer keeps stale contents.
    // Applied by codegen to every host-wrapper launch that takes thread_dimms.
    __host__ inline dim3 grid_host_clamp_threads(const void *kernel_fn, dim3 requested) {
        cudaFuncAttributes _attr;
        if (cudaFuncGetAttributes(&_attr, kernel_fn) != cudaSuccess) {
            cudaGetLastError();  // swallow -- fall back to the requested dims
            return requested;
        }
        unsigned _cap = (_attr.maxThreadsPerBlock > 0) ? (unsigned)_attr.maxThreadsPerBlock : requested.x;
        if (requested.x > _cap) requested.x = _cap;
        return requested;
    }

    
    template <typename T> __host__ __device__ constexpr size_t GRID_LINALG_NVIDIA_MAX_HELPER_BYTES();
    const int NUM_JOINTS = 7;
    const int NUM_POS = 7;
    const int NUM_VEL = 7;
    const int NUM_BODIES = 7;
    const int SECOND_ORDER_COORDS = 7;
    const int SECOND_ORDER_TENSOR_SIZE = 1372;
    const int Q_QD_U_STRIDE = 21;
    // h_q_qd_u / h_q_qd_qdd input ABI (PUBLISHED — docs: user_guide/concepts/input_output_abi):
    // three NUM_POS-wide slots per timestep (stride Q_QD_U_STRIDE = 3*NUM_POS):
    //   q at +GRID_Q_OFFSET | qd at +GRID_QD_OFFSET | u (or qdd) at +GRID_U_OFFSET.
    // qd/u/qdd are passed at nq width; floating base: nv live values in the LEADING
    // slots + one trailing pad each. Matrix/gradient OUTPUTS are nv-wide. Do NOT pack
    // tightly: on a floating base nq > nv, so a tight u lands at nq+nv while kernels
    // read 2*nq — in-bounds and silently wrong. Fixed base (nq == nv) cannot expose this.
    const int GRID_Q_OFFSET = 0;
    const int GRID_QD_OFFSET = 7;
    const int GRID_U_OFFSET = 14;
    const int GRID_QDD_OFFSET = 14;
    const int NUM_EES = 1;
    const int TOPOLOGY_HELPERS_COUNT = 0;
    const int DYNAMICS_XI_T_COUNT = 504;
    const int XHOM_T_COUNT = 144;
    const int DXHOM_T_COUNT = 112;
    const int D2XHOM_T_COUNT = 112;
    const int GRID_INVERSE_DYNAMICS_GRADIENT_USES_GLOBAL_TEMP = 0;
    const int GRID_INVERSE_DYNAMICS_GRADIENT_USES_WORKSPACE_ANY_TIER = 1;
    const int GRID_INVERSE_DYNAMICS_REGRESSOR_GRADIENT_USES_WORKSPACE_ANY_TIER = 1;
    const int GRID_FORWARD_DYNAMICS_GRADIENT_USES_GLOBAL_TEMP = 0;
    const int GRID_FORWARD_DYNAMICS_GRADIENT_USES_WORKSPACE_ANY_TIER = 1;
    const int GRID_INVERSE_DYNAMICS_GRADIENT_USES_DA_DF_SPILL = 0;
    const int GRID_FORWARD_DYNAMICS_GRADIENT_USES_DA_DF_SPILL = 0;
    const int GRID_INTEGRATOR_USES_WORKSPACE = 1;
    const int GRID_INTEGRATOR_GRADIENT_USES_WORKSPACE = 1;
    const int GRID_INTEGRATOR_GRADIENT_USES_DA_DF_SPILL = 0;
    const int GRID_GENERATES_IDSVA_SO_BODY_FRAME = 0;
    const int GRID_GENERATES_FDSVA_SO = 0;
    const int GRID_GENERATES_D2EE = 0;
    const int GRID_IDSVA_SO_USES_GLOBAL_OUTPUT = 0;
    const int GRID_FDSVA_SO_USES_GLOBAL_TENSORS = 0;
    const int GRID_FDSVA_SO_USES_WORKSPACE_TEMP = 0;
    const int GRID_FDSVA_SO_USES_WORKSPACE_ANY_TIER = 1;
    const int GRID_END_EFFECTOR_POSE_HESSIAN_USES_WORKSPACE_TEMP = 0;
    const int GRID_END_EFFECTOR_POSE_HESSIAN_USES_WORKSPACE_D2XHOM = 0;
    const int GRID_END_EFFECTOR_POSE_HESSIAN_USES_WORKSPACE_TEMP_ANY = 0;
    const int GRID_END_EFFECTOR_POSE_HESSIAN_SHARED_TIER_VALUE = 0;
    const int GRID_END_EFFECTOR_POSE_GRADIENT_USES_WORKSPACE_TEMP = 0;
    const int GRID_END_EFFECTOR_POSE_GRADIENT_USES_WORKSPACE_TEMP_ANY = 1;
    const int GRID_END_EFFECTOR_POSE_GRADIENT_USES_WORKSPACE_DXHOM = 0;
    const int GRID_END_EFFECTOR_POSE_GRADIENT_SHARED_TIER_VALUE = 0;
    const int GRID_DCCRBA_USES_WORKSPACE_TEMP = 0;
    const int GRID_OSC_INERTIA_USES_WORKSPACE = 0;
    const int GRID_INVERSE_DYNAMICS_GRADIENT_SHARED_TIER_VALUE = 0;
    const int GRID_FORWARD_DYNAMICS_GRADIENT_SHARED_TIER_VALUE = 0;
    const int ID_DU_TEMP_SPILL_START = 336;
    const int ID_DU_TEMP_SPILL_END = 1260;
    const int ID_DU_TEMP_SPILL_COUNT = 924;
    const int INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_COUNT = 708;
    const int MINV_DYNAMIC_SHARED_MEM_COUNT = 1235;
    const int FORWARD_DYNAMICS_DYNAMIC_SHARED_MEM_COUNT = 1256;
    const int INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_COUNT = 2479;
    const int FORWARD_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_COUNT = 2535;
    const int INTEGRATOR_DYNAMIC_SHARED_MEM_COUNT = 1333;
    const int INTEGRATOR_DU_DYNAMIC_SHARED_MEM_COUNT = 3643;
    const int ABA_DYNAMIC_SHARED_MEM_COUNT = 1604;
    const int CRBA_SHARED_MEM_COUNT = 869;
    const int ID_DU_MAX_SHARED_MEM_COUNT = 2479;
    const int FD_DU_MAX_SHARED_MEM_COUNT = 2535;
    const int END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_COUNT = 197;
    const int END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_COUNT = 391;
    const int END_EFFECTOR_POSE_HESSIAN_DYNAMIC_SHARED_MEM_COUNT = 755;
    const int IDSVA_SO_DYNAMIC_SHARED_MEM_COUNT = 5649;
    const int FDSVA_SO_DYNAMIC_SHARED_MEM_COUNT = 7182;
    const int MAX_PERF_LEVEL_THREADS = 352;
    
    // Resource-tier API (v2.0): each emitted kernel/_device/_inner takes a
    // `RESOURCE_TIER` template parameter that picks the (launch_bounds, smem,
    // register-footprint) profile. TIER_SHARED is the default and is the
    // current-best perf; TIER_LITE keeps the same launch_bounds but reduces
    // smem footprint (some intermediates moved to workspace global mem);
    // TIER_MINIMAL drops launch_bounds to 1024 for maximum block-size flexibility
    // at the cost of register slack. Inline-CUDA power users with tight outer
    // kernels pick LITE/MINIMAL to fit GRiD primitives in their resource budget.
    constexpr int TIER_SHARED    = 0;
    constexpr int TIER_LITE    = 1;
    constexpr int TIER_MINIMAL = 2;
    
    // Compile-time override for the default RESOURCE_TIER baked into every
    // emitted kernel template. Defaults to TIER_SHARED so existing callsites
    // (kernel<T>, host wrappers that call kernel<T><<<...>>>) keep their
    // current best-perf semantics. Bench harness sets this via
    // -DGRID_DEFAULT_RESOURCE_TIER=TIER_LITE (or TIER_MINIMAL) to sweep
    // per-tier perf without modifying host-wrapper template signatures.
    #ifndef GRID_DEFAULT_RESOURCE_TIER
    #define GRID_DEFAULT_RESOURCE_TIER TIER_SHARED
    #endif
    
    // Per-tier launch_bounds upper-bound (= max threads per block nvcc must
    // budget registers for). sm_120 has 65536 regs/block; nvcc enforces
    // regs_per_thread * max_threads <= regs_per_block, so a larger max_threads
    // directly caps regs_per_thread. PERF=SUGGESTED keeps current best perf;
    // LITE=min(2*SUGGESTED, 768) gives ~85 regs/thread cap (mid-budget);
    // MINIMAL=1024 gives ~64 regs/thread cap (maximum block-size flexibility).
    template <int TIER> __host__ __device__ constexpr int tier_max_threads() {
        return (TIER == TIER_MINIMAL) ? 1024
             : (TIER == TIER_LITE)    ? ((MAX_PERF_LEVEL_THREADS * 2 < 768) ? MAX_PERF_LEVEL_THREADS * 2 : 768)
             :                          MAX_PERF_LEVEL_THREADS;
    }
    
    // ─── A1b baked launch config (single source of truth) ───────────────
    // Autotuned per-algo {resource tier, threads-per-block} for THIS robot
    // + base, baked from config/launch_configs/KUKAiiwa14/rtx5090_sm120.json (fixed, profile=host).
    // GRiD kernels are single-block + thread-count-invariant, so (tier,threads)
    // is a pure PERFORMANCE choice; HOST launchers / python-jax-torch bindings
    // default their launch config from grid::launch_cfg<GRID_ALGO_*>. An algo
    // with no autotuned entry (or a robot/GPU with no launch_configs file)
    // falls back to the conservative (GRID_DEFAULT_RESOURCE_TIER,
    // MAX_PERF_LEVEL_THREADS) default -> un-tuned robots are unaffected.
    enum GridAlgo {
        GRID_ALGO_INVERSE_DYNAMICS,
        GRID_ALGO_MINV,
        GRID_ALGO_FORWARD_DYNAMICS,
        GRID_ALGO_ABA,
        GRID_ALGO_CRBA,
        GRID_ALGO_INVERSE_DYNAMICS_GRADIENT,
        GRID_ALGO_FORWARD_DYNAMICS_GRADIENT,
        GRID_ALGO_END_EFFECTOR_POSE,
        GRID_ALGO_END_EFFECTOR_POSE_GRADIENT,
        GRID_ALGO_END_EFFECTOR_POSE_HESSIAN,
        GRID_ALGO_IDSVA_SO,
        GRID_ALGO_IDSVA_SO_BODY_FRAME,
        GRID_ALGO_IDSVA_SO_WORLD_FRAME,
        GRID_ALGO_FDSVA_SO,
        GRID_ALGO_INTEGRATOR,
        GRID_ALGO_INTEGRATOR_GRADIENT,
        GRID_ALGO_INTEGRATOR_WITH_GRADIENT,
        GRID_ALGO_F_EXT_GRADIENT,
        GRID_ALGO_F_EXT_GRADIENT_DQ,
        GRID_ALGO_INVERSE_DYNAMICS_REGRESSOR,
        GRID_ALGO_FORWARD_DYNAMICS_PARAMETER_GRADIENT,
        GRID_ALGO_KINETIC_ENERGY_REGRESSOR,
        GRID_ALGO_POTENTIAL_ENERGY_REGRESSOR,
        GRID_ALGO_FRAME_JACOBIAN,
        GRID_ALGO_FRAME_JACOBIAN_DOT,
        GRID_ALGO_OSC_INERTIA,
        GRID_ALGO_GENERALIZED_GRAVITY,
        GRID_ALGO_NONLINEAR_EFFECTS,
        GRID_ALGO_ENERGY,
        GRID_ALGO_COM,
        GRID_ALGO_CCRBA,
        GRID_ALGO_CORIOLIS_MATRIX,
        GRID_ALGO_DCCRBA,
        GRID_ALGO_CMM_TIME_VARIATION,
        GRID_ALGO_INVERSE_DYNAMICS_REGRESSOR_GRADIENT,
        GRID_ALGO_COUNT
    };
    // Primary template = conservative fallback (matches the historical default).
    template <int ALGO> struct launch_cfg {
        static constexpr int TIER    = GRID_DEFAULT_RESOURCE_TIER;
        static constexpr int THREADS = MAX_PERF_LEVEL_THREADS;
    };
    template <> struct launch_cfg<GRID_ALGO_INVERSE_DYNAMICS> { static constexpr int TIER = TIER_MINIMAL; static constexpr int THREADS = ((128) < tier_max_threads<TIER_MINIMAL>()) ? (128) : tier_max_threads<TIER_MINIMAL>(); };
    template <> struct launch_cfg<GRID_ALGO_MINV> { static constexpr int TIER = TIER_LITE; static constexpr int THREADS = ((128) < tier_max_threads<TIER_LITE>()) ? (128) : tier_max_threads<TIER_LITE>(); };
    template <> struct launch_cfg<GRID_ALGO_FORWARD_DYNAMICS> { static constexpr int TIER = TIER_SHARED; static constexpr int THREADS = ((128) < tier_max_threads<TIER_SHARED>()) ? (128) : tier_max_threads<TIER_SHARED>(); };
    template <> struct launch_cfg<GRID_ALGO_ABA> { static constexpr int TIER = TIER_SHARED; static constexpr int THREADS = ((128) < tier_max_threads<TIER_SHARED>()) ? (128) : tier_max_threads<TIER_SHARED>(); };
    template <> struct launch_cfg<GRID_ALGO_CRBA> { static constexpr int TIER = TIER_SHARED; static constexpr int THREADS = ((96) < tier_max_threads<TIER_SHARED>()) ? (96) : tier_max_threads<TIER_SHARED>(); };
    template <> struct launch_cfg<GRID_ALGO_INVERSE_DYNAMICS_GRADIENT> { static constexpr int TIER = TIER_SHARED; static constexpr int THREADS = ((128) < tier_max_threads<TIER_SHARED>()) ? (128) : tier_max_threads<TIER_SHARED>(); };
    template <> struct launch_cfg<GRID_ALGO_FORWARD_DYNAMICS_GRADIENT> { static constexpr int TIER = TIER_SHARED; static constexpr int THREADS = ((128) < tier_max_threads<TIER_SHARED>()) ? (128) : tier_max_threads<TIER_SHARED>(); };
    template <> struct launch_cfg<GRID_ALGO_END_EFFECTOR_POSE> { static constexpr int TIER = TIER_LITE; static constexpr int THREADS = ((128) < tier_max_threads<TIER_LITE>()) ? (128) : tier_max_threads<TIER_LITE>(); };
    template <> struct launch_cfg<GRID_ALGO_END_EFFECTOR_POSE_GRADIENT> { static constexpr int TIER = TIER_MINIMAL; static constexpr int THREADS = ((320) < tier_max_threads<TIER_MINIMAL>()) ? (320) : tier_max_threads<TIER_MINIMAL>(); };
    template <> struct launch_cfg<GRID_ALGO_END_EFFECTOR_POSE_HESSIAN> { static constexpr int TIER = TIER_MINIMAL; static constexpr int THREADS = ((224) < tier_max_threads<TIER_MINIMAL>()) ? (224) : tier_max_threads<TIER_MINIMAL>(); };
    template <> struct launch_cfg<GRID_ALGO_IDSVA_SO> { static constexpr int TIER = TIER_SHARED; static constexpr int THREADS = ((320) < tier_max_threads<TIER_SHARED>()) ? (320) : tier_max_threads<TIER_SHARED>(); };
    template <> struct launch_cfg<GRID_ALGO_IDSVA_SO_BODY_FRAME> { static constexpr int TIER = TIER_LITE; static constexpr int THREADS = ((352) < tier_max_threads<TIER_LITE>()) ? (352) : tier_max_threads<TIER_LITE>(); };
    template <> struct launch_cfg<GRID_ALGO_IDSVA_SO_WORLD_FRAME> { static constexpr int TIER = TIER_LITE; static constexpr int THREADS = ((80) < tier_max_threads<TIER_LITE>()) ? (80) : tier_max_threads<TIER_LITE>(); };
    template <> struct launch_cfg<GRID_ALGO_FDSVA_SO> { static constexpr int TIER = TIER_SHARED; static constexpr int THREADS = ((192) < tier_max_threads<TIER_SHARED>()) ? (192) : tier_max_threads<TIER_SHARED>(); };
    template <> struct launch_cfg<GRID_ALGO_INTEGRATOR> { static constexpr int TIER = TIER_MINIMAL; static constexpr int THREADS = ((128) < tier_max_threads<TIER_MINIMAL>()) ? (128) : tier_max_threads<TIER_MINIMAL>(); };
    template <> struct launch_cfg<GRID_ALGO_INTEGRATOR_GRADIENT> { static constexpr int TIER = TIER_MINIMAL; static constexpr int THREADS = ((128) < tier_max_threads<TIER_MINIMAL>()) ? (128) : tier_max_threads<TIER_MINIMAL>(); };
    template <> struct launch_cfg<GRID_ALGO_INTEGRATOR_WITH_GRADIENT> { static constexpr int TIER = TIER_MINIMAL; static constexpr int THREADS = ((128) < tier_max_threads<TIER_MINIMAL>()) ? (128) : tier_max_threads<TIER_MINIMAL>(); };
    template <> struct launch_cfg<GRID_ALGO_F_EXT_GRADIENT> { static constexpr int TIER = TIER_SHARED; static constexpr int THREADS = ((128) < tier_max_threads<TIER_SHARED>()) ? (128) : tier_max_threads<TIER_SHARED>(); };
    template <> struct launch_cfg<GRID_ALGO_F_EXT_GRADIENT_DQ> { static constexpr int TIER = TIER_LITE; static constexpr int THREADS = ((224) < tier_max_threads<TIER_LITE>()) ? (224) : tier_max_threads<TIER_LITE>(); };
    template <> struct launch_cfg<GRID_ALGO_INVERSE_DYNAMICS_REGRESSOR> { static constexpr int TIER = TIER_SHARED; static constexpr int THREADS = ((128) < tier_max_threads<TIER_SHARED>()) ? (128) : tier_max_threads<TIER_SHARED>(); };
    template <> struct launch_cfg<GRID_ALGO_FORWARD_DYNAMICS_PARAMETER_GRADIENT> { static constexpr int TIER = TIER_SHARED; static constexpr int THREADS = ((112) < tier_max_threads<TIER_SHARED>()) ? (112) : tier_max_threads<TIER_SHARED>(); };
    template <> struct launch_cfg<GRID_ALGO_KINETIC_ENERGY_REGRESSOR> { static constexpr int TIER = TIER_MINIMAL; static constexpr int THREADS = ((128) < tier_max_threads<TIER_MINIMAL>()) ? (128) : tier_max_threads<TIER_MINIMAL>(); };
    template <> struct launch_cfg<GRID_ALGO_POTENTIAL_ENERGY_REGRESSOR> { static constexpr int TIER = TIER_MINIMAL; static constexpr int THREADS = ((112) < tier_max_threads<TIER_MINIMAL>()) ? (112) : tier_max_threads<TIER_MINIMAL>(); };
    template <> struct launch_cfg<GRID_ALGO_FRAME_JACOBIAN> { static constexpr int TIER = TIER_SHARED; static constexpr int THREADS = ((128) < tier_max_threads<TIER_SHARED>()) ? (128) : tier_max_threads<TIER_SHARED>(); };
    template <> struct launch_cfg<GRID_ALGO_FRAME_JACOBIAN_DOT> { static constexpr int TIER = TIER_MINIMAL; static constexpr int THREADS = ((128) < tier_max_threads<TIER_MINIMAL>()) ? (128) : tier_max_threads<TIER_MINIMAL>(); };
    template <> struct launch_cfg<GRID_ALGO_OSC_INERTIA> { static constexpr int TIER = TIER_SHARED; static constexpr int THREADS = ((128) < tier_max_threads<TIER_SHARED>()) ? (128) : tier_max_threads<TIER_SHARED>(); };
    template <> struct launch_cfg<GRID_ALGO_GENERALIZED_GRAVITY> { static constexpr int TIER = TIER_MINIMAL; static constexpr int THREADS = ((128) < tier_max_threads<TIER_MINIMAL>()) ? (128) : tier_max_threads<TIER_MINIMAL>(); };
    template <> struct launch_cfg<GRID_ALGO_NONLINEAR_EFFECTS> { static constexpr int TIER = TIER_MINIMAL; static constexpr int THREADS = ((112) < tier_max_threads<TIER_MINIMAL>()) ? (112) : tier_max_threads<TIER_MINIMAL>(); };
    template <> struct launch_cfg<GRID_ALGO_ENERGY> { static constexpr int TIER = TIER_SHARED; static constexpr int THREADS = ((320) < tier_max_threads<TIER_SHARED>()) ? (320) : tier_max_threads<TIER_SHARED>(); };
    template <> struct launch_cfg<GRID_ALGO_COM> { static constexpr int TIER = TIER_SHARED; static constexpr int THREADS = ((160) < tier_max_threads<TIER_SHARED>()) ? (160) : tier_max_threads<TIER_SHARED>(); };
    template <> struct launch_cfg<GRID_ALGO_CCRBA> { static constexpr int TIER = TIER_SHARED; static constexpr int THREADS = ((128) < tier_max_threads<TIER_SHARED>()) ? (128) : tier_max_threads<TIER_SHARED>(); };
    template <> struct launch_cfg<GRID_ALGO_CORIOLIS_MATRIX> { static constexpr int TIER = TIER_LITE; static constexpr int THREADS = ((320) < tier_max_threads<TIER_LITE>()) ? (320) : tier_max_threads<TIER_LITE>(); };
    template <> struct launch_cfg<GRID_ALGO_DCCRBA> { static constexpr int TIER = TIER_MINIMAL; static constexpr int THREADS = ((192) < tier_max_threads<TIER_MINIMAL>()) ? (192) : tier_max_threads<TIER_MINIMAL>(); };
    template <> struct launch_cfg<GRID_ALGO_CMM_TIME_VARIATION> { static constexpr int TIER = TIER_SHARED; static constexpr int THREADS = ((112) < tier_max_threads<TIER_SHARED>()) ? (112) : tier_max_threads<TIER_SHARED>(); };
    
    #define GRID_GENERATED_NUM_JOINTS 7
    #define GRID_GENERATED_NUM_EES 1
    
    template <typename T> __host__ __device__ constexpr size_t INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES() { return grid_shared_arena_bytes<T>(700, TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); }
    template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ constexpr size_t INVERSE_DYNAMICS_REGRESSOR_DYNAMIC_SHARED_MEM_BYTES() { return (TIER == TIER_SHARED) ? grid_shared_arena_bytes<T>(1183, TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) : (TIER == TIER_LITE) ? grid_shared_arena_bytes<T>(1183, TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) : grid_shared_arena_bytes<T>(693, TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); }
    template <int TIER> __host__ __device__ constexpr bool INVERSE_DYNAMICS_REGRESSOR_Y_IN_SMEM() { return (TIER == TIER_SHARED) ? true : (TIER == TIER_LITE) ? true : false; }
    template <typename T> __host__ __device__ constexpr size_t KINETIC_ENERGY_REGRESSOR_DYNAMIC_SHARED_MEM_BYTES() { return grid_shared_arena_bytes<T>(756, TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); }
    template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ constexpr size_t CORIOLIS_MATRIX_DYNAMIC_SHARED_MEM_BYTES() { return (TIER == TIER_SHARED) ? grid_shared_arena_bytes<T>(2247, TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) : (TIER == TIER_LITE) ? grid_shared_arena_bytes<T>(2247, TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) : grid_shared_arena_bytes<T>(518, TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); }
    template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ constexpr size_t FORWARD_DYNAMICS_PARAMETER_GRADIENT_DYNAMIC_SHARED_MEM_BYTES() { return (TIER == TIER_SHARED) ? grid_shared_arena_bytes<T>(2361, TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) : (TIER == TIER_LITE) ? grid_shared_arena_bytes<T>(2361, TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) : grid_shared_arena_bytes<T>(1871, TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); }
    template <int TIER> __host__ __device__ constexpr bool FD_PARAMETER_GRADIENT_Y_IN_SMEM() { return (TIER == TIER_SHARED) ? true : (TIER == TIER_LITE) ? true : false; }
    template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ constexpr size_t F_EXT_GRADIENT_DYNAMIC_SHARED_MEM_BYTES() { return (TIER == TIER_SHARED) ? grid_shared_arena_bytes<T>(1815, TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) : (TIER == TIER_LITE) ? grid_shared_arena_bytes<T>(1815, TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) : grid_shared_arena_bytes<T>(1064, TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); }
    template <int TIER> __host__ __device__ constexpr bool F_EXT_GRADIENT_DQDD_IN_SMEM() { return (TIER == TIER_SHARED) ? true : (TIER == TIER_LITE) ? true : false; }
    template <int TIER> __host__ __device__ constexpr bool F_EXT_GRADIENT_DTAU_IN_SMEM() { return (TIER == TIER_SHARED) ? true : (TIER == TIER_LITE) ? true : false; }
    template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ constexpr size_t F_EXT_GRADIENT_DQ_DYNAMIC_SHARED_MEM_BYTES() { return (TIER == TIER_SHARED) ? grid_shared_arena_bytes<T>(525, TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) : (TIER == TIER_LITE) ? grid_shared_arena_bytes<T>(525, TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) : grid_shared_arena_bytes<T>(525, TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); }
    template <int TIER> __host__ __device__ constexpr bool F_EXT_GRADIENT_DQ_SLAB_IN_SMEM() { return (TIER == TIER_SHARED) ? true : (TIER == TIER_LITE) ? true : false; }
    template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ constexpr size_t MINV_DYNAMIC_SHARED_MEM_BYTES() { return (TIER == TIER_SHARED) ? grid_shared_arena_bytes<T>(1227, TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) : (TIER == TIER_LITE) ? grid_shared_arena_bytes<T>(1227, TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) : grid_shared_arena_bytes<T>(933, TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); }
    template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ constexpr size_t FORWARD_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES() { return (TIER == TIER_SHARED) ? grid_shared_arena_bytes<T>(1248, TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) : (TIER == TIER_LITE) ? grid_shared_arena_bytes<T>(1248, TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) : grid_shared_arena_bytes<T>(954, TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); }
    template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ constexpr size_t INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES() { return (TIER == TIER_SHARED) ? grid_shared_arena_bytes<T>(2471, TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) : (TIER == TIER_LITE) ? grid_shared_arena_bytes<T>(2471, TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) : grid_shared_arena_bytes<T>(749, TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); }
    template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ constexpr size_t INVERSE_DYNAMICS_REGRESSOR_GRADIENT_DYNAMIC_SHARED_MEM_BYTES() { return (TIER == TIER_SHARED) ? grid_shared_arena_bytes<T>(2471, TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) : (TIER == TIER_LITE) ? grid_shared_arena_bytes<T>(2471, TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) : grid_shared_arena_bytes<T>(749, TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); }
    template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ constexpr size_t FORWARD_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES() { return (TIER == TIER_SHARED) ? grid_shared_arena_bytes<T>(2527, TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) : (TIER == TIER_LITE) ? grid_shared_arena_bytes<T>(2527, TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) : grid_shared_arena_bytes<T>(658, TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); }
    template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ constexpr size_t INTEGRATOR_DYNAMIC_SHARED_MEM_BYTES() { return (TIER == TIER_SHARED) ? grid_shared_arena_bytes<T>(1325, TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) : (TIER == TIER_LITE) ? grid_shared_arena_bytes<T>(1325, TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) : grid_shared_arena_bytes<T>(1031, TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); }
    template <int TIER> __host__ __device__ constexpr bool INTEGRATOR_MINV_F_IN_SMEM() { return (TIER == TIER_SHARED) ? true : (TIER == TIER_LITE) ? true : false; }
    template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ constexpr size_t INTEGRATOR_DU_DYNAMIC_SHARED_MEM_BYTES() { return (TIER == TIER_SHARED) ? grid_shared_arena_bytes<T>(3635, TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) : (TIER == TIER_LITE) ? grid_shared_arena_bytes<T>(3635, TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) : grid_shared_arena_bytes<T>(1031, TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); }
    template <int TIER> __host__ __device__ constexpr bool INTEGRATOR_DU_D_QDD_IN_SMEM() { return (TIER == TIER_SHARED) ? true : (TIER == TIER_LITE) ? true : false; }
    template <int TIER> __host__ __device__ constexpr bool INTEGRATOR_DU_DAB_IN_SMEM() { return (TIER == TIER_SHARED) ? true : (TIER == TIER_LITE) ? true : false; }
    template <int TIER> __host__ __device__ constexpr int INTEGRATOR_DU_INNER_LEVEL() { return (TIER == TIER_SHARED) ? 0 : (TIER == TIER_LITE) ? 0 : 2; }
    template <typename T> __host__ __device__ constexpr size_t GRID_INTEGRATOR_GRADIENT_DAB_OFFSET_BYTES() { return sizeof(T) * static_cast<size_t>(588); }
    template <typename T> __host__ __device__ constexpr size_t GRID_INTEGRATOR_GRADIENT_INNER_OFFSET_BYTES() { return sizeof(T) * static_cast<size_t>(882); }
    template <typename T> __host__ __device__ constexpr size_t INVERSE_DYNAMICS_DEVICE_DYNAMIC_SHARED_MEM_BYTES() { return grid_shared_arena_bytes<T>(672, TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); }
    template <typename T> __host__ __device__ constexpr size_t MINV_DEVICE_DYNAMIC_SHARED_MEM_BYTES() { return grid_shared_arena_bytes<T>(1171, TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); }
    template <typename T> __host__ __device__ constexpr size_t FORWARD_DYNAMICS_DEVICE_DYNAMIC_SHARED_MEM_BYTES() { return grid_shared_arena_bytes<T>(1220, TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); }
    template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ constexpr size_t FORWARD_DYNAMICS_DEVICE_INLINE_SMEM_BYTES() {
        return (TIER == TIER_SHARED)
            ? grid_shared_arena_bytes<T>(1220, TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>())
            : grid_shared_arena_bytes<T>(504, TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());
    }
    template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ constexpr size_t FORWARD_DYNAMICS_DEVICE_INLINE_WORKSPACE_BYTES() { return (TIER == TIER_SHARED) ? static_cast<size_t>(0) : sizeof(T) * static_cast<size_t>(716); }
    template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ constexpr size_t ABA_DYNAMIC_SHARED_MEM_BYTES() { return (TIER == TIER_SHARED) ? grid_shared_arena_bytes<T>(1596, TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) : (TIER == TIER_LITE) ? grid_shared_arena_bytes<T>(1596, TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) : grid_shared_arena_bytes<T>(616, TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); }
    template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ constexpr size_t CRBA_DYNAMIC_SHARED_MEM_BYTES() { return (TIER == TIER_SHARED) ? grid_shared_arena_bytes<T>(861, TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) : (TIER == TIER_LITE) ? grid_shared_arena_bytes<T>(861, TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) : grid_shared_arena_bytes<T>(518, TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); }
    template <typename T> __host__ __device__ constexpr size_t GRID_EE_LINALG_SHARED_BYTES() { return static_cast<size_t>(0); }
    template <typename T> __host__ __device__ constexpr size_t POTENTIAL_ENERGY_REGRESSOR_DYNAMIC_SHARED_MEM_BYTES() { return grid_shared_arena_bytes<T>(333, TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()); }
    template <typename T> __host__ __device__ constexpr size_t END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_BYTES() { return grid_shared_arena_bytes<T>(189, TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()); }
    template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ constexpr size_t END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES() { return (TIER == TIER_SHARED) ? grid_shared_arena_bytes<T>(383, TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()) : (TIER == TIER_LITE) ? grid_shared_arena_bytes<T>(383, TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()) : grid_shared_arena_bytes<T>(151, TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()); }
    template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ constexpr size_t END_EFFECTOR_POSE_HESSIAN_DYNAMIC_SHARED_MEM_BYTES() { return (TIER == TIER_SHARED) ? grid_shared_arena_bytes<T>(747, TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()) : (TIER == TIER_LITE) ? grid_shared_arena_bytes<T>(747, TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()) : grid_shared_arena_bytes<T>(747, TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()); }
    template <typename T> __host__ __device__ constexpr size_t INVERSE_DYNAMICS_BIAS_DYNAMIC_SHARED_MEM_BYTES() { return grid_shared_arena_bytes<T>(700, TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); }
    template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ constexpr size_t COM_DYNAMIC_SHARED_MEM_BYTES() { return (TIER == TIER_SHARED) ? grid_shared_arena_bytes<T>(960, TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()) : (TIER == TIER_LITE) ? grid_shared_arena_bytes<T>(960, TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()) : grid_shared_arena_bytes<T>(666, TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()); }
    template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ constexpr size_t CCRBA_DYNAMIC_SHARED_MEM_BYTES() { return (TIER == TIER_SHARED) ? grid_shared_arena_bytes<T>(991, TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()) : (TIER == TIER_LITE) ? grid_shared_arena_bytes<T>(991, TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()) : grid_shared_arena_bytes<T>(697, TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()); }
    template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ constexpr size_t ENERGY_DYNAMIC_SHARED_MEM_BYTES() { return (TIER == TIER_SHARED) ? grid_shared_arena_bytes<T>(946, TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()) : (TIER == TIER_LITE) ? grid_shared_arena_bytes<T>(946, TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()) : grid_shared_arena_bytes<T>(652, TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()); }
    template <int TIER> __host__ __device__ constexpr bool COM_J_IN_SMEM() { return (TIER == TIER_SHARED) ? true : (TIER == TIER_LITE) ? true : false; }
    template <int TIER> __host__ __device__ constexpr bool CCRBA_J_IN_SMEM() { return (TIER == TIER_SHARED) ? true : (TIER == TIER_LITE) ? true : false; }
    template <int TIER> __host__ __device__ constexpr bool ENERGY_J_IN_SMEM() { return (TIER == TIER_SHARED) ? true : (TIER == TIER_LITE) ? true : false; }
    template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ constexpr size_t CMM_TIME_VARIATION_DYNAMIC_SHARED_MEM_BYTES() { return (TIER == TIER_SHARED) ? grid_shared_arena_bytes<T>(1027, TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()) : (TIER == TIER_LITE) ? grid_shared_arena_bytes<T>(1027, TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()) : grid_shared_arena_bytes<T>(733, TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()); }
    template <int TIER> __host__ __device__ constexpr bool CMM_J_IN_SMEM() { return (TIER == TIER_SHARED) ? true : (TIER == TIER_LITE) ? true : false; }
    template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ constexpr size_t DCCRBA_DYNAMIC_SHARED_MEM_BYTES() { return (TIER == TIER_SHARED) ? grid_shared_arena_bytes<T>(1272, TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()) : (TIER == TIER_LITE) ? grid_shared_arena_bytes<T>(1272, TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()) : grid_shared_arena_bytes<T>(684, TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()); }
    template <int TIER> __host__ __device__ constexpr bool DCCRBA_OUTPUT_IN_SMEM() { return (TIER == TIER_SHARED) ? true : (TIER == TIER_LITE) ? true : false; }
    template <int TIER> __host__ __device__ constexpr bool DCCRBA_J_IN_SMEM() { return (TIER == TIER_SHARED) ? true : (TIER == TIER_LITE) ? true : false; }
    template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ constexpr size_t IDSVA_SO_BODY_FRAME_DYNAMIC_SHARED_MEM_BYTES() { return (TIER == TIER_SHARED) ? grid_shared_arena_bytes<T>(5641, TOPOLOGY_HELPERS_COUNT) : (TIER == TIER_LITE) ? grid_shared_arena_bytes<T>(5641, TOPOLOGY_HELPERS_COUNT) : grid_shared_arena_bytes<T>(525, TOPOLOGY_HELPERS_COUNT); }
    template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ constexpr size_t IDSVA_SO_WORLD_FRAME_DYNAMIC_SHARED_MEM_BYTES() { return (TIER == TIER_SHARED) ? grid_shared_arena_bytes<T>(5641, TOPOLOGY_HELPERS_COUNT) : (TIER == TIER_LITE) ? grid_shared_arena_bytes<T>(5641, TOPOLOGY_HELPERS_COUNT) : grid_shared_arena_bytes<T>(525, TOPOLOGY_HELPERS_COUNT); }
    template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ constexpr size_t FDSVA_SO_DYNAMIC_SHARED_MEM_BYTES() { return (TIER == TIER_SHARED) ? grid_shared_arena_bytes<T>(7174, TOPOLOGY_HELPERS_COUNT) : (TIER == TIER_LITE) ? grid_shared_arena_bytes<T>(7174, TOPOLOGY_HELPERS_COUNT) : grid_shared_arena_bytes<T>(686, TOPOLOGY_HELPERS_COUNT); }
    // Per-tier scratch sizes for fdsva_so_contract (inline-CUDA users only — the host launchers always use TIER_SHARED).
    // At TIER_SHARED the 4*NV^3 inner scratch lives in s_temp; at TIER_LITE/MINIMAL it moves to d_workspace, freeing shared memory for the caller's outer kernel.
    // fdsva_so_contract scratch sizing, keyed on the INNER's placement choice
    // (SCRATCH_IN_SMEM) rather than a tier — the inner decides placement, the
    // caller sizes both arenas from these. FDSVA_SO_SCRATCH_IN_SMEM<TIER>()
    // gives the placement codegen assigned to each tier for THIS robot.
    template <typename T, bool SCRATCH_IN_SMEM = true> __host__ __device__ constexpr size_t FDSVA_SO_INNER_SMEM_BYTES() { return SCRATCH_IN_SMEM ? sizeof(T) * static_cast<size_t>(1372) : static_cast<size_t>(0); }
    template <typename T, bool SCRATCH_IN_SMEM = true> __host__ __device__ constexpr size_t FDSVA_SO_INNER_WORKSPACE_BYTES() { return SCRATCH_IN_SMEM ? static_cast<size_t>(0) : sizeof(T) * static_cast<size_t>(1372); }
    // Per-robot tier->placement map: contraction scratch stays in smem at any rung that doesn't set use_workspace_temp.
    template <int TIER> __host__ __device__ constexpr bool FDSVA_SO_SCRATCH_IN_SMEM() { return (TIER == TIER_SHARED) ? true : (TIER == TIER_LITE) ? true : false; }
    // Inner-controlled placement API (design rollout): each inline inner is keyed on a
    // placement bool and decides arena pointers itself. *_INNER_{SMEM,WORKSPACE}_BYTES<T, IN_SMEM>
    // give the two arena sizes; *_<...>_IN_SMEM<TIER>() give the per-robot tier->placement
    // map codegen assigned (multiple tiers may share a placement on small robots).
    // --- minv_inner (F-region) ---
    template <typename T, bool F_IN_SMEM = true> __host__ __device__ constexpr size_t MINV_INNER_SMEM_BYTES() { return sizeof(T) * static_cast<size_t>(373 + 294 * (F_IN_SMEM ? 1 : 0)); }
    template <typename T, bool F_IN_SMEM = true> __host__ __device__ constexpr size_t MINV_INNER_WORKSPACE_BYTES() { return F_IN_SMEM ? static_cast<size_t>(0) : sizeof(T) * static_cast<size_t>(294); }
    template <int TIER> __host__ __device__ constexpr bool MINV_F_IN_SMEM() { return (TIER == TIER_SHARED) ? true : (TIER == TIER_LITE) ? true : false; }
    // --- forward_dynamics_inner (internal Minv F-region) ---
    template <typename T, bool MINV_F_IN_SMEM = true> __host__ __device__ constexpr size_t FD_INNER_SMEM_BYTES() { return MINV_F_IN_SMEM ? sizeof(T) * static_cast<size_t>(716) : sizeof(T) * static_cast<size_t>(422); }
    template <typename T, bool MINV_F_IN_SMEM = true> __host__ __device__ constexpr size_t FD_INNER_WORKSPACE_BYTES() { return MINV_F_IN_SMEM ? static_cast<size_t>(0) : sizeof(T) * static_cast<size_t>(294); }
    template <int TIER> __host__ __device__ constexpr bool FD_MINV_F_IN_SMEM() { return (TIER == TIER_SHARED) ? true : (TIER == TIER_LITE) ? true : false; }
    // --- integrator_inner (forwards the FD inner's Minv F-region lever) ---
    template <typename T, bool MINV_F_IN_SMEM = true> __host__ __device__ constexpr size_t INTEGRATOR_INNER_SMEM_BYTES() { return MINV_F_IN_SMEM ? sizeof(T) * static_cast<size_t>(716) : sizeof(T) * static_cast<size_t>(422); }
    template <typename T, bool MINV_F_IN_SMEM = true> __host__ __device__ constexpr size_t INTEGRATOR_INNER_WORKSPACE_BYTES() { return MINV_F_IN_SMEM ? static_cast<size_t>(0) : sizeof(T) * static_cast<size_t>(294); }
    // --- aba_inner (scratch band, surgical-spill ladder) ---
    // Levels: 0=full (smem), 1=surgical (hot smem + cold d_cold), 2=workspace (whole band global).
    // TEMP_IN_SMEM is false only at the level-2 (workspace) rung; COLD_IN_SMEM is false only at the level-1 (surgical) rung.
    template <typename T, bool TEMP_IN_SMEM = true> __host__ __device__ constexpr size_t ABA_INNER_SMEM_BYTES() { return TEMP_IN_SMEM ? sizeof(T) * static_cast<size_t>(980) : static_cast<size_t>(0); }
    template <typename T, bool TEMP_IN_SMEM = true> __host__ __device__ constexpr size_t ABA_INNER_WORKSPACE_BYTES() { return TEMP_IN_SMEM ? static_cast<size_t>(0) : sizeof(T) * static_cast<size_t>(980); }
    template <typename T> __host__ __device__ constexpr size_t ABA_INNER_COLD_BYTES() { return sizeof(T) * static_cast<size_t>(294); }
    template <int TIER> __host__ __device__ constexpr bool ABA_TEMP_IN_SMEM() { return (TIER == TIER_SHARED) ? true : (TIER == TIER_LITE) ? true : false; }
    template <int TIER> __host__ __device__ constexpr bool ABA_COLD_IN_SMEM() { return (TIER == TIER_SHARED) ? true : (TIER == TIER_LITE) ? true : true; }
    // --- crba_inner (scratch band) ---
    template <typename T, bool TEMP_IN_SMEM = true> __host__ __device__ constexpr size_t CRBA_INNER_SMEM_BYTES() { return TEMP_IN_SMEM ? sizeof(T) * static_cast<size_t>(294) : static_cast<size_t>(0); }
    template <typename T, bool TEMP_IN_SMEM = true> __host__ __device__ constexpr size_t CRBA_INNER_WORKSPACE_BYTES() { return TEMP_IN_SMEM ? static_cast<size_t>(0) : sizeof(T) * static_cast<size_t>(294); }
    template <int TIER> __host__ __device__ constexpr bool CRBA_TEMP_IN_SMEM() { return (TIER == TIER_SHARED) ? true : (TIER == TIER_LITE) ? true : false; }
    template <int TIER> __host__ __device__ constexpr bool CRBA_M_IN_SMEM() { return (TIER == TIER_SHARED) ? true : (TIER == TIER_LITE) ? true : false; }
    // --- end_effector_pose_gradient_inner (chain workspace) ---
    template <typename T, bool TEMP_IN_SMEM = true> __host__ __device__ constexpr size_t EE_GRAD_INNER_SMEM_BYTES() { return TEMP_IN_SMEM ? sizeof(T) * static_cast<size_t>(190) : static_cast<size_t>(0); }
    template <typename T, bool TEMP_IN_SMEM = true> __host__ __device__ constexpr size_t EE_GRAD_INNER_WORKSPACE_BYTES() { return TEMP_IN_SMEM ? static_cast<size_t>(0) : sizeof(T) * static_cast<size_t>(190); }
    template <int TIER> __host__ __device__ constexpr bool EE_GRAD_TEMP_IN_SMEM() { return (TIER == TIER_SHARED) ? true : (TIER == TIER_LITE) ? true : false; }
    // --- end_effector_pose_hessian_inner (large nv^2 end_effector_pose_hessian output) ---
    // Per-tier placement of the d2ee inner's OUTPUT s_end_effector_pose_hessian: true => smem, false => d_workspace (which the kernel sets to d_end_effector_pose_hessian directly).
    template <int TIER> __host__ __device__ constexpr bool D2EE_OUT_IN_SMEM() { return (TIER == TIER_SHARED) ? true : (TIER == TIER_LITE) ? true : true; }
    // Per-tier sizes for forward_dynamics_gradient_device (inline-CUDA users only). At TIER_SHARED the temp scratch arena lives in s_temp; at TIER_LITE/MINIMAL it moves to d_workspace, freeing roughly 1722*sizeof(T) bytes of smem.
    template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ constexpr size_t FORWARD_DYNAMICS_GRADIENT_DEVICE_INLINE_SMEM_BYTES() {
        return (TIER == TIER_SHARED)
            ? grid_shared_arena_bytes<T>(2506, TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>())
            : grid_shared_arena_bytes<T>(784, TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());
    }
    template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ constexpr size_t FORWARD_DYNAMICS_GRADIENT_DEVICE_INLINE_WORKSPACE_BYTES() { return (TIER == TIER_SHARED) ? static_cast<size_t>(0) : sizeof(T) * static_cast<size_t>(1722); }
    // Per-tier sizes for end_effector_pose_hessian_device (inline-CUDA users only). At TIER_SHARED the smem arena keeps only the FD scratch + s_Xhom; at TIER_LITE/MINIMAL the device contract is unchanged (smem arena is the same -- the caller-provided s_end_effector_pose_hessian is what shifts), and the inner writes its 294*sizeof(T) output bytes to d_workspace instead.
    template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ constexpr size_t END_EFFECTOR_POSE_HESSIAN_DEVICE_INLINE_SMEM_BYTES() {
        return grid_shared_arena_bytes<T>(404, TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>());
    }
    template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ constexpr size_t END_EFFECTOR_POSE_HESSIAN_DEVICE_INLINE_WORKSPACE_BYTES() { return (TIER == TIER_SHARED) ? static_cast<size_t>(0) : sizeof(T) * static_cast<size_t>(294); }
    // Per-tier sizes for inverse_dynamics_gradient_device (inline-CUDA users only). At TIER_SHARED temp lives in s_temp; at TIER_LITE/MINIMAL it moves to d_workspace, freeing 1722*sizeof(T) bytes of smem.
    template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ constexpr size_t INVERSE_DYNAMICS_GRADIENT_DEVICE_INLINE_SMEM_BYTES() {
        return (TIER == TIER_SHARED)
            ? grid_shared_arena_bytes<T>(2352, TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>())
            : grid_shared_arena_bytes<T>(630, TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());
    }
    template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ constexpr size_t INVERSE_DYNAMICS_GRADIENT_DEVICE_INLINE_WORKSPACE_BYTES() { return (TIER == TIER_SHARED) ? static_cast<size_t>(0) : sizeof(T) * static_cast<size_t>(1722); }
    // Per-tier sizes for idsva_so_device (inline-CUDA users only). At TIER_SHARED temp lives in s_temp; at TIER_LITE/MINIMAL it moves to d_workspace, freeing 3744*sizeof(T) bytes of smem. Frame picked at codegen time: body_frame.
    template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ constexpr size_t IDSVA_SO_DEVICE_INLINE_SMEM_BYTES() {
        return (TIER == TIER_SHARED)
            ? grid_shared_arena_bytes<T>(4248, TOPOLOGY_HELPERS_COUNT)
            : grid_shared_arena_bytes<T>(504, TOPOLOGY_HELPERS_COUNT);
    }
    template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ constexpr size_t IDSVA_SO_DEVICE_INLINE_WORKSPACE_BYTES() { return (TIER == TIER_SHARED) ? static_cast<size_t>(0) : sizeof(T) * static_cast<size_t>(3744); }
    template <typename T> __host__ __device__ constexpr size_t GRID_GRAD_WORKSPACE_BYTES_PER_TIMESTEP() { return sizeof(T) * static_cast<size_t>(2604); }
    template <typename T> __host__ __device__ constexpr size_t GRID_SO_WORKSPACE_BYTES_PER_TIMESTEP() { return sizeof(T) * static_cast<size_t>(3744); }
    template <typename T> __host__ __device__ constexpr size_t GRID_FDSVA_SO_SPILL_BYTES_PER_TIMESTEP() { return sizeof(T) * static_cast<size_t>(147); }
    template <typename T> __host__ __device__ constexpr size_t GRID_FDSVA_SO_SPILL_OFFSET_BYTES() { return GRID_GRAD_WORKSPACE_BYTES_PER_TIMESTEP<T>() + GRID_SO_WORKSPACE_BYTES_PER_TIMESTEP<T>(); }
    template <typename T> __host__ __device__ constexpr size_t GRID_WORKSPACE_BYTES_PER_TIMESTEP() { return GRID_GRAD_WORKSPACE_BYTES_PER_TIMESTEP<T>() + GRID_SO_WORKSPACE_BYTES_PER_TIMESTEP<T>() + GRID_FDSVA_SO_SPILL_BYTES_PER_TIMESTEP<T>(); }
    template <typename T> __host__ __device__ inline gridSharedTier GRID_INVERSE_DYNAMICS_GRADIENT_SHARED_TIER() { return static_cast<gridSharedTier>(GRID_INVERSE_DYNAMICS_GRADIENT_SHARED_TIER_VALUE); }
    template <typename T> __host__ __device__ inline gridSharedTier GRID_FORWARD_DYNAMICS_GRADIENT_SHARED_TIER() { return static_cast<gridSharedTier>(GRID_FORWARD_DYNAMICS_GRADIENT_SHARED_TIER_VALUE); }
    template <typename T> __host__ __device__ constexpr size_t GRID_SO_WORKSPACE_TEMP_OFFSET_BYTES() { return GRID_GRAD_WORKSPACE_BYTES_PER_TIMESTEP<T>(); }
    template <typename T> __host__ __device__ constexpr size_t GRID_DCCRBA_J_OFFSET_BYTES() { return GRID_SO_WORKSPACE_TEMP_OFFSET_BYTES<T>() + sizeof(T) * static_cast<size_t>(294); }
    template <typename T> __host__ __device__ constexpr size_t GRID_MINV_F_WORKSPACE_OFFSET_BYTES() { return static_cast<size_t>(0); }
    template <typename T> __host__ __device__ constexpr size_t GRID_ABA_COLD_OFFSET_BYTES() { return static_cast<size_t>(0); }
    template <typename T> __host__ __device__ constexpr size_t GRID_END_EFFECTOR_POSE_HESSIAN_WORKSPACE_TEMP_OFFSET_BYTES() { return static_cast<size_t>(0); }
    template <typename T> __host__ __device__ constexpr size_t GRID_END_EFFECTOR_POSE_HESSIAN_WORKSPACE_D2XHOM_OFFSET_BYTES() { return static_cast<size_t>(0); }
    template <typename T> __host__ __device__ constexpr size_t GRID_END_EFFECTOR_POSE_HESSIAN_WORKSPACE_D2EETEMP_OFFSET_BYTES() { return static_cast<size_t>(0); }
    template <typename T> __host__ __device__ constexpr size_t GRID_END_EFFECTOR_POSE_GRADIENT_WORKSPACE_DXHOM_OFFSET_BYTES() { return GRID_SO_WORKSPACE_TEMP_OFFSET_BYTES<T>(); }
    template <typename T> __host__ __device__ constexpr size_t GRID_END_EFFECTOR_POSE_GRADIENT_WORKSPACE_TEMP_OFFSET_BYTES() { return GRID_END_EFFECTOR_POSE_GRADIENT_WORKSPACE_DXHOM_OFFSET_BYTES<T>() + (GRID_END_EFFECTOR_POSE_GRADIENT_USES_WORKSPACE_DXHOM ? sizeof(T) * static_cast<size_t>(DXHOM_T_COUNT) : 0); }
    template <typename T> __host__ __device__ inline bool grid_selected_shared_memory_fits() { return INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>() <= GRID_CUDA_TARGET_SHARED_MEM_BYTES && FORWARD_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>() <= GRID_CUDA_TARGET_SHARED_MEM_BYTES && (!GRID_GENERATES_D2EE || END_EFFECTOR_POSE_HESSIAN_DYNAMIC_SHARED_MEM_BYTES<T>() <= GRID_CUDA_TARGET_SHARED_MEM_BYTES) && (!GRID_GENERATES_IDSVA_SO_BODY_FRAME || IDSVA_SO_BODY_FRAME_DYNAMIC_SHARED_MEM_BYTES<T>() <= GRID_CUDA_TARGET_SHARED_MEM_BYTES) && (!GRID_GENERATES_FDSVA_SO || FDSVA_SO_DYNAMIC_SHARED_MEM_BYTES<T>() <= GRID_CUDA_TARGET_SHARED_MEM_BYTES); }
    // __forceinline__ used throughout the xhom helper chain so ptxas folds these into the
    // inner kernels at all opt levels. For fixed-base the body of grid_q_index_affects_joint is
    // the trivial `q_index == joint_id` check that pre-GLASS callsites used directly.
    __host__ __device__ __forceinline__ bool grid_q_index_affects_joint(const int q_index, const int joint_id) {
        return q_index == joint_id;
    }
    __host__ __device__ __forceinline__ int grid_d2xhom_offset(const int q_index_i, [[maybe_unused]] const int q_index_j) {
        return 16 * q_index_i;
    }
    template <typename T>
    __device__ __forceinline__ const T *grid_xhom_or_dxhom_ptr(const T *s_Xhom, const T *s_dXhom, const int q_index, const int joint_id) {
        return grid_q_index_affects_joint(q_index, joint_id) ? &s_dXhom[16 * q_index] : &s_Xhom[16 * joint_id];
    }
    template <typename T>
    __device__ __forceinline__ const T *grid_xhom_or_dxhom_or_d2xhom_ptr(const T *s_Xhom, const T *s_dXhom, const T *s_d2Xhom, const int q_index_i, const int q_index_j, const int joint_id) {
        const bool i_affects = grid_q_index_affects_joint(q_index_i, joint_id);
        const bool j_affects = grid_q_index_affects_joint(q_index_j, joint_id);
        if (i_affects && j_affects) { return &s_d2Xhom[grid_d2xhom_offset(q_index_i, q_index_j)]; }
        if (i_affects) { return &s_dXhom[16 * q_index_i]; }
        if (j_affects) { return &s_dXhom[16 * q_index_j]; }
        return &s_Xhom[16 * joint_id];
    }
    template <typename T, bool USE_DA_DF_SPILL>
    __device__ inline T *grid_id_du_temp_ptr(T *s_temp, T *d_temp_spill, int index) {
        if (!USE_DA_DF_SPILL) { return &s_temp[index]; }
        if (index >= ID_DU_TEMP_SPILL_START && index < ID_DU_TEMP_SPILL_END) {
            return &d_temp_spill[index - ID_DU_TEMP_SPILL_START];
        }
        if (index >= ID_DU_TEMP_SPILL_END) {
            return &s_temp[index - ID_DU_TEMP_SPILL_COUNT];
        }
        return &s_temp[index];
    }
    
    // Define custom structs
    template <typename T>
    struct robotModel {
        T *d_XImats;
        int *d_topology_helpers;
    };
    struct grid_device_pool_t;  // defined with the allocator below
    template <typename T, gridDataKind KIND = GRID_DATA_ALL>
    struct gridData {
        grid_device_pool_t *pool;  // the allocator this arena was carved from (W04-B B1/K1); the default pool unless init_gridData_checked was given one
        // GPU INPUTS
        T *d_q_qd_u;
        T *d_q_qd;
        T *d_q;
        T *d_f_ext;
        // CPU INPUTS
        T *h_q_qd_u;
        T *h_q_qd;
        T *h_q;
        T *h_f_ext;
        // GPU OUTPUTS
        T *d_c;
        T *d_Minv;
        T *d_qdd;
        T *d_M;
        T *d_dc_du;
        T *d_df_du;
        T *d_dtau_dfext;
        T *d_dqdd_dfext;
        T *d_f_ext_gradient_dq;  // -dJ^T/dq = d(inverse_dynamics_gradient)/dfext, nv*6NB*nv (both base modes)
        T *d_Y;          // inverse_dynamics_regressor (tau = Y . pi), nv*10NB
        T *d_dY_dx;      // inverse_dynamics_regressor_gradient (dY/dq | dY/dqd), 2*nv*nv*10NB
        T *d_dqdd_dpi;   // forward_dynamics_parameter_gradient (-Minv . Y), nv*10NB
        T *d_ke_regressor;   // kinetic_energy_regressor (KE = y_KE . pi), 10NB
        T *d_pe_regressor;   // potential_energy_regressor (PE = y_PE . pi), 10NB
        T *d_coriolis;       // coriolis_matrix C(q,qd), nv*nv
        T *d_dccrba;             // dccrba dA_dq[:,k,m], 6*nv*nv
        T *d_cmm_time_variation; // cmm_time_variation Adot, 6*nv
        T *d_end_effector_pose;
        T *d_end_effector_pose_gradient;
        T *d_end_effector_pose_hessian;
        T *d_frame_jacobian;       // frame_jacobian (6 x NUM_VEL)
        T *d_frame_jacobian_dot;   // frame_jacobian_dot (6 x NUM_VEL)
        T *d_osc_inertia;          // osc_inertia Lambda (6 x 6)
        T *d_eePose;               // end_effector_pose_runtime (6 = [xyz;rpy])
        T *d_eePoseGrad;           // end_effector_pose_gradient_runtime (6 x NUM_VEL)
        T *d_eepose_runtime_offset; // runtime 4x4 col-major SE(3) tool/tip transform (target frame)
        unsigned char *d_workspace;
        int workspace_timestep_slots;
        T *d_idsva_so;
        T *d_df2;
        T *d_x_kp1;
        T *d_dAB;
        T *d_com;
        T *d_ccrba;
        T *d_energy;
        // CPU OUTPUTS
        T *h_c;
        T *h_Minv;
        T *h_qdd;
        T *h_M;
        T *h_dc_du;
        T *h_df_du;
        T *h_dtau_dfext;
        T *h_dqdd_dfext;
        T *h_f_ext_gradient_dq;  // -dJ^T/dq, nv*6NB*nv (both base modes)
        T *h_Y;
        T *h_dY_dx;
        T *h_dqdd_dpi;
        T *h_ke_regressor;
        T *h_pe_regressor;
        T *h_coriolis;
        T *h_dccrba;
        T *h_cmm_time_variation;
        T *h_end_effector_pose;
        T *h_end_effector_pose_gradient;
        T *h_end_effector_pose_hessian;
        T *h_frame_jacobian;
        T *h_frame_jacobian_dot;
        T *h_osc_inertia;
        T *h_eePose;
        T *h_eePoseGrad;
        T *h_idsva_so;
        T *h_df2;
        T *h_x_kp1;
        T *h_dAB;
        T *h_com;
        T *h_ccrba;
        T *h_energy;
    };
    /**
     * GLASS linear algebra helpers (SIMT only) — consumed from the top-level GLASS (vendor_glass=False)
     *
     */
    
    // vendor_glass=False: GLASS is NOT vendored. The consumer's include path
    // must provide the top-level glass.cuh (included in this header's prelude);
    // the generator was run against GLASS revision 8ce68a29bceb30c7764c9391d517a182d061697d.
    namespace glass = ::glass;
    
    /**
     * Linear algebra wrappers (SIMT GLASS)
     *
     */
    // SIMT-only linalg. The `glass_nvidia_smem` parameter on each wrapper is
    // retained for caller compatibility and is always ignored; the stub
    // `GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()` below returns 0 so
    // shared-memory arena calculations continue to compile unchanged.
    
    template <typename T>
    __host__ __device__ constexpr size_t GRID_LINALG_NVIDIA_MAX_HELPER_BYTES() {
        return static_cast<size_t>(0);
    }
    
    template <typename T, int M, int N, int K, bool TRANSPOSE_B = false, bool ROW_MAJOR_A = false, bool ROW_MAJOR_B = false, bool ROW_MAJOR_C = false>
    __device__ void grid_linalg_gemm(const T *A, const T *B, T *C, T alpha, T beta, unsigned char *glass_nvidia_smem = nullptr) {
        (void)glass_nvidia_smem;
        T *A_mut = const_cast<T *>(A);
        T *B_mut = const_cast<T *>(B);
        // GLASS v2 gemm: contraction is the LAST template dim, so the old (M,N,K)
        // contraction-in-the-middle maps to <M,K,N>. A row-major operand equals its
        // col-major transpose (ROW_MAJOR_A -> TRANSPOSE_A; ROW_MAJOR_B XORs into
        // TRANSPOSE_B); ROW_MAJOR_C is the only surviving per-operand layout flag.
        glass::gemm<T, M, K, N, ROW_MAJOR_A, (TRANSPOSE_B != ROW_MAJOR_B), ROW_MAJOR_C>(alpha, A_mut, B_mut, beta, C);
        __syncthreads();
    }
    
    template <typename T, int M, int N, bool TRANSPOSE = false, bool ROW_MAJOR_A = false>
    __device__ void grid_linalg_gemv(const T *A, const T *x, T *y, T alpha, T beta, unsigned char *glass_nvidia_smem = nullptr) {
        (void)glass_nvidia_smem;
        T *A_mut = const_cast<T *>(A);
        T *x_mut = const_cast<T *>(x);
        // GLASS v2: gemv keeps its per-operand ROW_MAJOR flag (TRANSPOSE selects the math
        // op A*x vs A^T*x, independent of storage); the redundant gemv_ex was removed.
        glass::gemv<T, M, N, TRANSPOSE, ROW_MAJOR_A>(alpha, A_mut, x_mut, beta, y);
        __syncthreads();
    }
    
    template <typename T, int M, int N, int ROW_STRIDE>
    __device__ void grid_linalg_row_strided_gemv(const T *A, const T *x, T *y, T alpha, T beta, unsigned char *glass_nvidia_smem = nullptr) {
        (void)glass_nvidia_smem;
        glass::gemv_strided<T, M, N, ROW_STRIDE>(alpha, A, x, beta, y);
        __syncthreads();
    }
    
    template <typename T, int M, int N, int K, int A_RS, int B_RS>
    __device__ void grid_linalg_row_strided_gemm(const T *A, const T *B, T *C, T alpha, T beta, unsigned char *glass_nvidia_smem = nullptr) {
        (void)glass_nvidia_smem;
        glass::gemm_strided<T, M, K, N, A_RS, B_RS>(alpha, A, B, beta, C);
        __syncthreads();
    }
    
    template <typename T, int N, int S1, int S2>
    __device__ T grid_linalg_dot_strided(const T *vec1, const T *vec2) {
        return glass::dot_strided<T, N, S1, S2>(vec1, vec2);
    }
    
    // Segmented (batched) row-strided GEMV: `segments` independent M x N GEMVs in one
    // block-cooperative pass, base offsets per segment via the descriptor arrays. With
    // FUSE_SCALED_ADD, folds a per-segment y += S*scalar add into the single y store.
    template <typename T, int M, int N, int ROW_STRIDE = M, bool FUSE_SCALED_ADD = false>
    __device__ void grid_linalg_segmented_row_strided_gemv(unsigned int segments, const int *seg_a_off, const int *seg_x_off, const int *seg_y_off, const T *A, const T *x, T *y, T alpha, T beta, const int *seg_s_off = nullptr, const T *S = nullptr, const T *scalar = nullptr, unsigned char *glass_nvidia_smem = nullptr) {
        (void)glass_nvidia_smem;
        glass::gemv_segmented<T, M, N, ROW_STRIDE, FUSE_SCALED_ADD>(segments, seg_a_off, seg_x_off, seg_y_off, A, x, y, alpha, beta, seg_s_off, S, scalar);
        __syncthreads();
    }
    
    // Indexed batched DIMxDIM (col-major) GEMM: C[c_idx[p]] = A[a_idx[p]] * B[b_idx[p]]
    // over a flat in-chain index list (e.g. compacted (ancestor,ee) pairs).
    template <typename T, int DIM = 4>
    __device__ void grid_linalg_indexed_batched_gemm(unsigned int pairs, const int *a_idx, const int *b_idx, const int *c_idx, const T *A_base, const T *B_base, T *C_base, unsigned char *glass_nvidia_smem = nullptr) {
        (void)glass_nvidia_smem;
        glass::gemm_batched_indexed<T, DIM>(pairs, a_idx, b_idx, c_idx, A_base, B_base, C_base);
        __syncthreads();
    }
    
    // Coalesced block-cooperative strided dot: same value as grid_linalg_dot_strided but
    // the whole block cooperates on ONE dot with transposed/tiled iteration so consecutive
    // threads hit consecutive global addresses (fast when the operand is L2-pinned global).
    // Writes the scalar to *out (valid after the trailing barrier); needs ceil(blockDim/32)
    // T of s_scratch. NOT a drop-in for grid_linalg_dot_strided (that one is per-thread).
    template <typename T, int N, int SX = 1, int SY = 1>
    __device__ void grid_linalg_dot_strided_coalesced(const T *x, const T *y, T *out, T *s_scratch) {
        glass::dot_strided_coalesced<T, N, SX, SY>(x, y, out, s_scratch);
        __syncthreads();
    }
    
    /**
     * Compute the dot product between two vectors
     *
     * Notes:
     *   Assumes computed by a single thread
     *
     * @param vec1 is the first vector of length N with stride S1
     * @param vec2 is the second vector of length N with stride S2
     * @return the resulting final value
     */
    template <typename T, int N, int S1, int S2>
    __device__
    T dot_prod(const T *vec1, const T *vec2) {
        return glass::dot_strided<T, N, S1, S2>(vec1, vec2);
    }

    /**
     * Compute the dot product between two vectors
     *
     * Notes:
     *   Assumes computed by a single thread
     *
     * @param vec1 is the first vector of length N with stride S1
     * @param vec2 is the second vector of length N with stride S2
     * @return the resulting final value
     */
    template <typename T, int N, int S1, int S2>
    __device__
    T dot_prod(T *vec1, const T *vec2) {
        return glass::dot_strided<T, N, S1, S2>(vec1, vec2);
    }

    /**
     * Compute the dot product between two vectors
     *
     * Notes:
     *   Assumes computed by a single thread
     *
     * @param vec1 is the first vector of length N with stride S1
     * @param vec2 is the second vector of length N with stride S2
     * @return the resulting final value
     */
    template <typename T, int N, int S1, int S2>
    __device__
    T dot_prod(const T *vec1, T *vec2) {
        return glass::dot_strided<T, N, S1, S2>(vec1, vec2);
    }

    /**
     * Compute the dot product between two vectors
     *
     * Notes:
     *   Assumes computed by a single thread
     *
     * @param vec1 is the first vector of length N with stride S1
     * @param vec2 is the second vector of length N with stride S2
     * @return the resulting final value
     */
    template <typename T, int N, int S1, int S2>
    __device__
    T dot_prod(T *vec1, T *vec2) {
        return glass::dot_strided<T, N, S1, S2>(vec1, vec2);
    }

    /**
     * Generates the motion vector cross product matrix column 0
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     */
    template <typename T>
    __device__
    void mx0(T *s_vecX, const T *s_vec) {
        glass::thread::motion_cross_mul<T, 0, false>(static_cast<T>(1), s_vec, nullptr, static_cast<T>(0), s_vecX);
    }

    /**
     * Adds the motion vector cross product matrix column 0
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     */
    template <typename T>
    __device__
    void mx0_peq(T *s_vecX, const T *s_vec) {
        glass::thread::motion_cross_mul<T, 0, true>(static_cast<T>(1), s_vec, nullptr, static_cast<T>(1), s_vecX);
    }

    /**
     * Generates the motion vector cross product matrix column 0
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     * @param alpha is the scaling factor
     */
    template <typename T>
    __device__
    void mx0_scaled(T *s_vecX, const T *s_vec, const T alpha) {
        glass::thread::motion_cross_mul<T, 0, false>(alpha, s_vec, nullptr, static_cast<T>(0), s_vecX);
    }

    /**
     * Adds the motion vector cross product matrix column 0
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     * @param alpha is the scaling factor
     */
    template <typename T>
    __device__
    void mx0_peq_scaled(T *s_vecX, const T *s_vec, const T alpha) {
        glass::thread::motion_cross_mul<T, 0, true>(alpha, s_vec, nullptr, static_cast<T>(1), s_vecX);
    }

    /**
     * Generates the motion vector cross product matrix column 1
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     */
    template <typename T>
    __device__
    void mx1(T *s_vecX, const T *s_vec) {
        glass::thread::motion_cross_mul<T, 1, false>(static_cast<T>(1), s_vec, nullptr, static_cast<T>(0), s_vecX);
    }

    /**
     * Adds the motion vector cross product matrix column 1
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     */
    template <typename T>
    __device__
    void mx1_peq(T *s_vecX, const T *s_vec) {
        glass::thread::motion_cross_mul<T, 1, true>(static_cast<T>(1), s_vec, nullptr, static_cast<T>(1), s_vecX);
    }

    /**
     * Generates the motion vector cross product matrix column 1
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     * @param alpha is the scaling factor
     */
    template <typename T>
    __device__
    void mx1_scaled(T *s_vecX, const T *s_vec, const T alpha) {
        glass::thread::motion_cross_mul<T, 1, false>(alpha, s_vec, nullptr, static_cast<T>(0), s_vecX);
    }

    /**
     * Adds the motion vector cross product matrix column 1
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     * @param alpha is the scaling factor
     */
    template <typename T>
    __device__
    void mx1_peq_scaled(T *s_vecX, const T *s_vec, const T alpha) {
        glass::thread::motion_cross_mul<T, 1, true>(alpha, s_vec, nullptr, static_cast<T>(1), s_vecX);
    }

    /**
     * Generates the motion vector cross product matrix column 2
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     */
    template <typename T>
    __device__
    void mx2(T *s_vecX, const T *s_vec) {
        glass::thread::motion_cross_mul<T, 2, false>(static_cast<T>(1), s_vec, nullptr, static_cast<T>(0), s_vecX);
    }

    /**
     * Adds the motion vector cross product matrix column 2
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     */
    template <typename T>
    __device__
    void mx2_peq(T *s_vecX, const T *s_vec) {
        glass::thread::motion_cross_mul<T, 2, true>(static_cast<T>(1), s_vec, nullptr, static_cast<T>(1), s_vecX);
    }

    /**
     * Generates the motion vector cross product matrix column 2
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     * @param alpha is the scaling factor
     */
    template <typename T>
    __device__
    void mx2_scaled(T *s_vecX, const T *s_vec, const T alpha) {
        glass::thread::motion_cross_mul<T, 2, false>(alpha, s_vec, nullptr, static_cast<T>(0), s_vecX);
    }

    /**
     * Adds the motion vector cross product matrix column 2
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     * @param alpha is the scaling factor
     */
    template <typename T>
    __device__
    void mx2_peq_scaled(T *s_vecX, const T *s_vec, const T alpha) {
        glass::thread::motion_cross_mul<T, 2, true>(alpha, s_vec, nullptr, static_cast<T>(1), s_vecX);
    }

    /**
     * Generates the motion vector cross product matrix column 3
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     */
    template <typename T>
    __device__
    void mx3(T *s_vecX, const T *s_vec) {
        glass::thread::motion_cross_mul<T, 3, false>(static_cast<T>(1), s_vec, nullptr, static_cast<T>(0), s_vecX);
    }

    /**
     * Adds the motion vector cross product matrix column 3
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     */
    template <typename T>
    __device__
    void mx3_peq(T *s_vecX, const T *s_vec) {
        glass::thread::motion_cross_mul<T, 3, true>(static_cast<T>(1), s_vec, nullptr, static_cast<T>(1), s_vecX);
    }

    /**
     * Generates the motion vector cross product matrix column 3
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     * @param alpha is the scaling factor
     */
    template <typename T>
    __device__
    void mx3_scaled(T *s_vecX, const T *s_vec, const T alpha) {
        glass::thread::motion_cross_mul<T, 3, false>(alpha, s_vec, nullptr, static_cast<T>(0), s_vecX);
    }

    /**
     * Adds the motion vector cross product matrix column 3
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     * @param alpha is the scaling factor
     */
    template <typename T>
    __device__
    void mx3_peq_scaled(T *s_vecX, const T *s_vec, const T alpha) {
        glass::thread::motion_cross_mul<T, 3, true>(alpha, s_vec, nullptr, static_cast<T>(1), s_vecX);
    }

    /**
     * Generates the motion vector cross product matrix column 4
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     */
    template <typename T>
    __device__
    void mx4(T *s_vecX, const T *s_vec) {
        glass::thread::motion_cross_mul<T, 4, false>(static_cast<T>(1), s_vec, nullptr, static_cast<T>(0), s_vecX);
    }

    /**
     * Adds the motion vector cross product matrix column 4
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     */
    template <typename T>
    __device__
    void mx4_peq(T *s_vecX, const T *s_vec) {
        glass::thread::motion_cross_mul<T, 4, true>(static_cast<T>(1), s_vec, nullptr, static_cast<T>(1), s_vecX);
    }

    /**
     * Generates the motion vector cross product matrix column 4
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     * @param alpha is the scaling factor
     */
    template <typename T>
    __device__
    void mx4_scaled(T *s_vecX, const T *s_vec, const T alpha) {
        glass::thread::motion_cross_mul<T, 4, false>(alpha, s_vec, nullptr, static_cast<T>(0), s_vecX);
    }

    /**
     * Adds the motion vector cross product matrix column 4
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     * @param alpha is the scaling factor
     */
    template <typename T>
    __device__
    void mx4_peq_scaled(T *s_vecX, const T *s_vec, const T alpha) {
        glass::thread::motion_cross_mul<T, 4, true>(alpha, s_vec, nullptr, static_cast<T>(1), s_vecX);
    }

    /**
     * Generates the motion vector cross product matrix column 5
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     */
    template <typename T>
    __device__
    void mx5(T *s_vecX, const T *s_vec) {
        glass::thread::motion_cross_mul<T, 5, false>(static_cast<T>(1), s_vec, nullptr, static_cast<T>(0), s_vecX);
    }

    /**
     * Adds the motion vector cross product matrix column 5
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     */
    template <typename T>
    __device__
    void mx5_peq(T *s_vecX, const T *s_vec) {
        glass::thread::motion_cross_mul<T, 5, true>(static_cast<T>(1), s_vec, nullptr, static_cast<T>(1), s_vecX);
    }

    /**
     * Generates the motion vector cross product matrix column 5
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     * @param alpha is the scaling factor
     */
    template <typename T>
    __device__
    void mx5_scaled(T *s_vecX, const T *s_vec, const T alpha) {
        glass::thread::motion_cross_mul<T, 5, false>(alpha, s_vec, nullptr, static_cast<T>(0), s_vecX);
    }

    /**
     * Adds the motion vector cross product matrix column 5
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     * @param alpha is the scaling factor
     */
    template <typename T>
    __device__
    void mx5_peq_scaled(T *s_vecX, const T *s_vec, const T alpha) {
        glass::thread::motion_cross_mul<T, 5, true>(alpha, s_vec, nullptr, static_cast<T>(1), s_vecX);
    }

    /**
     * Generates the motion vector cross product matrix for a runtime selected column
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     */
    template <typename T>
    __device__
    void mxX(T *s_vecX, const T *s_vec, const int S_ind) {
        switch(S_ind){
            case 0: mx0<T>(s_vecX, s_vec); break;
            case 1: mx1<T>(s_vecX, s_vec); break;
            case 2: mx2<T>(s_vecX, s_vec); break;
            case 3: mx3<T>(s_vecX, s_vec); break;
            case 4: mx4<T>(s_vecX, s_vec); break;
            case 5: mx5<T>(s_vecX, s_vec); break;
        }
    }

    /**
     * Generates the motion vector cross product matrix for a runtime selected column
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     */
    template <typename T>
    __device__
    void mxX_peq(T *s_vecX, const T *s_vec, const int S_ind) {
        switch(S_ind){
            case 0: mx0_peq<T>(s_vecX, s_vec); break;
            case 1: mx1_peq<T>(s_vecX, s_vec); break;
            case 2: mx2_peq<T>(s_vecX, s_vec); break;
            case 3: mx3_peq<T>(s_vecX, s_vec); break;
            case 4: mx4_peq<T>(s_vecX, s_vec); break;
            case 5: mx5_peq<T>(s_vecX, s_vec); break;
        }
    }

    /**
     * Generates the motion vector cross product matrix for a runtime selected column
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     * @param alpha is the scaling factor
     */
    template <typename T>
    __device__
    void mxX_scaled(T *s_vecX, const T *s_vec, const T alpha, const int S_ind) {
        switch(S_ind){
            case 0: mx0_scaled<T>(s_vecX, s_vec, alpha); break;
            case 1: mx1_scaled<T>(s_vecX, s_vec, alpha); break;
            case 2: mx2_scaled<T>(s_vecX, s_vec, alpha); break;
            case 3: mx3_scaled<T>(s_vecX, s_vec, alpha); break;
            case 4: mx4_scaled<T>(s_vecX, s_vec, alpha); break;
            case 5: mx5_scaled<T>(s_vecX, s_vec, alpha); break;
        }
    }

    /**
     * Generates the motion vector cross product matrix for a runtime selected column
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     * @param alpha is the scaling factor
     */
    template <typename T>
    __device__
    void mxX_peq_scaled(T *s_vecX, const T *s_vec, const T alpha, const int S_ind) {
        switch(S_ind){
            case 0: mx0_peq_scaled<T>(s_vecX, s_vec, alpha); break;
            case 1: mx1_peq_scaled<T>(s_vecX, s_vec, alpha); break;
            case 2: mx2_peq_scaled<T>(s_vecX, s_vec, alpha); break;
            case 3: mx3_peq_scaled<T>(s_vecX, s_vec, alpha); break;
            case 4: mx4_peq_scaled<T>(s_vecX, s_vec, alpha); break;
            case 5: mx5_peq_scaled<T>(s_vecX, s_vec, alpha); break;
        }
    }

    /**
     * Generates the force cross product matrix and multiplies by the input vector
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_result is the result vector
     * @param s_fxVec is the fx vector
     * @param s_timesVec is the multipled vector
     */
    template <typename T>
    __device__
    void fx_times_v(T *s_result, const T *s_fxVec, const T *s_timesVec) {
        glass::thread::force_cross_mul<T, false>(static_cast<T>(1), s_fxVec, s_timesVec, static_cast<T>(0), s_result);
    }

    /**
     * Adds the force cross product matrix multiplied by the input vector
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_result is the result vector
     * @param s_fxVec is the fx vector
     * @param s_timesVec is the multipled vector
     */
    template <typename T>
    __device__
    void fx_times_v_peq(T *s_result, const T *s_fxVec, const T *s_timesVec) {
        glass::thread::force_cross_mul<T, true>(static_cast<T>(1), s_fxVec, s_timesVec, static_cast<T>(1), s_result);
    }

    /**
     * Generates the full motion vector cross product matrix (6x6, column-major)
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param dest is the destination 6x6 matrix
     * @param v is the source 6-vector
     */
    template <typename T>
    __device__
    void vcross(T *dest, T *v){
        glass::thread::motion_cross(v, dest);
    }

    /**
     * Compute the inverse force cross product matrix of a 6-vector, v Returns the entry at the index.
     *
     * Notes:
     *   ICRF is the operation defined such that v crf f = f icrf v
     *
     * @param index is the index of the result matirx to compute
     * @param v is the 6-vector to take the cross product matrix of
     */
    template <typename T>
    __device__
    T icrf(int index, T *v) {
        return glass::spatial_detail::force_cross_dual_entry<T>((uint32_t)(index % 6), (uint32_t)(index / 6), v);
    }

    /**
     * Compute the motion cross product matrix of a 6-vector, v Returns the entry at the index.
     *
     * Notes:
     *   The force cross product matrix is just the negative transpose of this matrix
     *
     * @param index is the index of the result matirx to compute
     * @param v is the 6-vector to take the cross product matrix of
     */
    template <typename T>
    __device__
    T crm(int index, T *v) {
        return glass::spatial_detail::motion_cross_entry<T>((uint32_t)(index % 6), (uint32_t)(index / 6), v);
    }

    /**
     * Compute the motion cross product multiplication of a 6-vector, v_crm, with a second 6-vector v
     *
     * @param index is the index of the result vector to compute
     * @param v_crm is the 6-vector to take the cross product matrix of
     * @param v is the 6-vector to multiply with v_crm
     */
    template <typename T>
    __device__
    T crm_mul(int index, T *v_crm, T *v) {
        return glass::spatial_detail::motion_cross_mul_row<T>((uint32_t)index, v_crm, v);
    }

    /**
     * Compute the inverse of a matrix (wraps glass::inv_dense)
     *
     * Notes:
     *   Block-cooperative Gauss-Jordan via GLASS.
     *   Both A and Ainv hold A^-1 on return (dual-output for caller compat).
     *   s_temp must hold at least 3*dimA elements.
     *
     * @param dimA is the matrix dimension
     * @param A is the original invertible matrix (overwritten with A^-1 on return)
     * @param Ainv is workspace; on return it also holds A^-1
     * @param s_temp is shared scratch of size >= 3*dimA
     */
    template <typename T>
    __device__
    void invert_matrix(uint32_t dimA, T *A, T *Ainv, T *s_temp) {
        glass::inv_dense<T>(dimA, A, Ainv, s_temp);
    }

    /**
     * Matrix multiplication helper function of AB
     *
     * @param index - the index of the result vector
     * @param A - pointer to the first matrix
     * @param B - pointer to the second matrix
     * @param dest - pointer to the destination matrix
     * @param num - 36 or 6 depending on the indexing scheme
     * @param t - true => multiply with the transpose of B
     */
    template <typename T>
    __device__
    void matmul(int index, T *A, T *B, T *dest, int num, bool t) {
        int cur = 36*((index/num)%NUM_BODIES);
        T *vec1 = &B[cur + (t*5+1)*(index%6)];
        T *vec2 = &A[6*(index/6)];
        dest[index] = dot_prod<T,6, 6, 1>(vec1, vec2);
    }

    /**
     * Matrix multiplication helper function where one of the matrices is tranposed.
     *
     * @param index - the index of the result vector
     * @param A - pointer to the first 6x6 matrix
     * @param B - pointer to the second 6x6 matrix
     * @param dest - pointer to the destination matrix
     * @param char trans_mat - a for A^TB, b for AB^T
     */
    template <typename T>
    __device__
    void matmul_trans(int index, T *A, T *B, T *dest, char trans_mat) {
        T *vec1;
        T *vec2;
        if (trans_mat == 'a'){
            vec1 = &A[6*(index%6)];
            vec2 = &B[6*(index/6)];
            dest[index] = dot_prod<T,6,1,1>(vec1, vec2);
        }
        if (trans_mat == 'b'){
            vec1 = &A[index%6];
            vec2 = &B[index/6];
            dest[index] = dot_prod<T,6,6,6>(vec1, vec2);
        }
    }

    //
    // Topology Helpers not needed!
    //
    template <typename T>
    __host__
    cudaError_t init_topology_helpers_checked(int **out, const char **failed_op = nullptr){ (void)failed_op; *out = nullptr; return cudaSuccess; }
    template <typename T>
    __host__
    int *init_topology_helpers(){return nullptr;}
    /**
     * Initializes the Xmats and Imats in GPU memory
     *
     * Notes:
     *   Memory order is X[0...N], I[0...N], Xhom[0...N]
     *
     * @return A pointer to the XI memory in the GPU
     */
    template <typename T>
    __host__
    cudaError_t init_XImats_checked(T **out, const char **failed_op = nullptr) {
        *out = nullptr;
        T *h_XImats = (T *)GRID_HOST_ALLOC(calloc(872,sizeof(T)));
        if (h_XImats == nullptr) { return grid_fail(failed_op, "calloc(h_XImats)", cudaErrorMemoryAllocation); }
        // X[0]
        h_XImats[0] = static_cast<T>(0);
        h_XImats[1] = static_cast<T>(0);
        h_XImats[2] = static_cast<T>(0);
        h_XImats[3] = static_cast<T>(0);
        h_XImats[4] = static_cast<T>(0);
        h_XImats[5] = static_cast<T>(0);
        h_XImats[6] = static_cast<T>(0);
        h_XImats[7] = static_cast<T>(0);
        h_XImats[8] = static_cast<T>(0);
        h_XImats[9] = static_cast<T>(0);
        h_XImats[10] = static_cast<T>(0);
        h_XImats[11] = static_cast<T>(0);
        h_XImats[12] = static_cast<T>(0);
        h_XImats[13] = static_cast<T>(0);
        h_XImats[14] = static_cast<T>(1.00000000000000);
        h_XImats[15] = static_cast<T>(0);
        h_XImats[16] = static_cast<T>(0);
        h_XImats[17] = static_cast<T>(0);
        h_XImats[18] = static_cast<T>(0);
        h_XImats[19] = static_cast<T>(0);
        h_XImats[20] = static_cast<T>(0);
        h_XImats[21] = static_cast<T>(0);
        h_XImats[22] = static_cast<T>(0);
        h_XImats[23] = static_cast<T>(0);
        h_XImats[24] = static_cast<T>(0);
        h_XImats[25] = static_cast<T>(0);
        h_XImats[26] = static_cast<T>(0);
        h_XImats[27] = static_cast<T>(0);
        h_XImats[28] = static_cast<T>(0);
        h_XImats[29] = static_cast<T>(0);
        h_XImats[30] = static_cast<T>(0);
        h_XImats[31] = static_cast<T>(0);
        h_XImats[32] = static_cast<T>(0);
        h_XImats[33] = static_cast<T>(0);
        h_XImats[34] = static_cast<T>(0);
        h_XImats[35] = static_cast<T>(1.00000000000000);
        // X[1]
        h_XImats[36] = static_cast<T>(0);
        h_XImats[37] = static_cast<T>(0);
        h_XImats[38] = static_cast<T>(-1.60982338570648e-15);
        h_XImats[39] = static_cast<T>(0);
        h_XImats[40] = static_cast<T>(0);
        h_XImats[41] = static_cast<T>(-0.202500000000000);
        h_XImats[42] = static_cast<T>(0);
        h_XImats[43] = static_cast<T>(0);
        h_XImats[44] = static_cast<T>(1.00000000000000);
        h_XImats[45] = static_cast<T>(0);
        h_XImats[46] = static_cast<T>(0);
        h_XImats[47] = static_cast<T>(0);
        h_XImats[48] = static_cast<T>(0);
        h_XImats[49] = static_cast<T>(0);
        h_XImats[50] = static_cast<T>(0);
        h_XImats[51] = static_cast<T>(0);
        h_XImats[52] = static_cast<T>(0);
        h_XImats[53] = static_cast<T>(0);
        h_XImats[54] = static_cast<T>(0);
        h_XImats[55] = static_cast<T>(0);
        h_XImats[56] = static_cast<T>(0);
        h_XImats[57] = static_cast<T>(0);
        h_XImats[58] = static_cast<T>(0);
        h_XImats[59] = static_cast<T>(-1.60982338570648e-15);
        h_XImats[60] = static_cast<T>(0);
        h_XImats[61] = static_cast<T>(0);
        h_XImats[62] = static_cast<T>(0);
        h_XImats[63] = static_cast<T>(0);
        h_XImats[64] = static_cast<T>(0);
        h_XImats[65] = static_cast<T>(1.00000000000000);
        h_XImats[66] = static_cast<T>(0);
        h_XImats[67] = static_cast<T>(0);
        h_XImats[68] = static_cast<T>(0);
        h_XImats[69] = static_cast<T>(0);
        h_XImats[70] = static_cast<T>(0);
        h_XImats[71] = static_cast<T>(0);
        // X[2]
        h_XImats[72] = static_cast<T>(0);
        h_XImats[73] = static_cast<T>(0);
        h_XImats[74] = static_cast<T>(1.00000000000000);
        h_XImats[75] = static_cast<T>(0);
        h_XImats[76] = static_cast<T>(0);
        h_XImats[77] = static_cast<T>(0);
        h_XImats[78] = static_cast<T>(0);
        h_XImats[79] = static_cast<T>(0);
        h_XImats[80] = static_cast<T>(0);
        h_XImats[81] = static_cast<T>(0);
        h_XImats[82] = static_cast<T>(0);
        h_XImats[83] = static_cast<T>(0);
        h_XImats[84] = static_cast<T>(0);
        h_XImats[85] = static_cast<T>(0);
        h_XImats[86] = static_cast<T>(0);
        h_XImats[87] = static_cast<T>(0);
        h_XImats[88] = static_cast<T>(0);
        h_XImats[89] = static_cast<T>(0);
        h_XImats[90] = static_cast<T>(0);
        h_XImats[91] = static_cast<T>(0);
        h_XImats[92] = static_cast<T>(0);
        h_XImats[93] = static_cast<T>(0);
        h_XImats[94] = static_cast<T>(0);
        h_XImats[95] = static_cast<T>(1.00000000000000);
        h_XImats[96] = static_cast<T>(0);
        h_XImats[97] = static_cast<T>(0);
        h_XImats[98] = static_cast<T>(0);
        h_XImats[99] = static_cast<T>(0);
        h_XImats[100] = static_cast<T>(0);
        h_XImats[101] = static_cast<T>(0);
        h_XImats[102] = static_cast<T>(0);
        h_XImats[103] = static_cast<T>(0);
        h_XImats[104] = static_cast<T>(0);
        h_XImats[105] = static_cast<T>(0);
        h_XImats[106] = static_cast<T>(0);
        h_XImats[107] = static_cast<T>(0);
        // X[3]
        h_XImats[108] = static_cast<T>(0);
        h_XImats[109] = static_cast<T>(0);
        h_XImats[110] = static_cast<T>(0);
        h_XImats[111] = static_cast<T>(0);
        h_XImats[112] = static_cast<T>(0);
        h_XImats[113] = static_cast<T>(0.215500000000000);
        h_XImats[114] = static_cast<T>(0);
        h_XImats[115] = static_cast<T>(0);
        h_XImats[116] = static_cast<T>(-1.00000000000000);
        h_XImats[117] = static_cast<T>(0);
        h_XImats[118] = static_cast<T>(0);
        h_XImats[119] = static_cast<T>(0);
        h_XImats[120] = static_cast<T>(0);
        h_XImats[121] = static_cast<T>(0);
        h_XImats[122] = static_cast<T>(0);
        h_XImats[123] = static_cast<T>(0);
        h_XImats[124] = static_cast<T>(0);
        h_XImats[125] = static_cast<T>(0);
        h_XImats[126] = static_cast<T>(0);
        h_XImats[127] = static_cast<T>(0);
        h_XImats[128] = static_cast<T>(0);
        h_XImats[129] = static_cast<T>(0);
        h_XImats[130] = static_cast<T>(0);
        h_XImats[131] = static_cast<T>(0);
        h_XImats[132] = static_cast<T>(0);
        h_XImats[133] = static_cast<T>(0);
        h_XImats[134] = static_cast<T>(0);
        h_XImats[135] = static_cast<T>(0);
        h_XImats[136] = static_cast<T>(0);
        h_XImats[137] = static_cast<T>(-1.00000000000000);
        h_XImats[138] = static_cast<T>(0);
        h_XImats[139] = static_cast<T>(0);
        h_XImats[140] = static_cast<T>(0);
        h_XImats[141] = static_cast<T>(0);
        h_XImats[142] = static_cast<T>(0);
        h_XImats[143] = static_cast<T>(0);
        // X[4]
        h_XImats[144] = static_cast<T>(0);
        h_XImats[145] = static_cast<T>(0);
        h_XImats[146] = static_cast<T>(0);
        h_XImats[147] = static_cast<T>(0);
        h_XImats[148] = static_cast<T>(0);
        h_XImats[149] = static_cast<T>(0);
        h_XImats[150] = static_cast<T>(0);
        h_XImats[151] = static_cast<T>(0);
        h_XImats[152] = static_cast<T>(1.00000000000000);
        h_XImats[153] = static_cast<T>(0);
        h_XImats[154] = static_cast<T>(0);
        h_XImats[155] = static_cast<T>(0);
        h_XImats[156] = static_cast<T>(0);
        h_XImats[157] = static_cast<T>(0);
        h_XImats[158] = static_cast<T>(0);
        h_XImats[159] = static_cast<T>(0);
        h_XImats[160] = static_cast<T>(0);
        h_XImats[161] = static_cast<T>(0);
        h_XImats[162] = static_cast<T>(0);
        h_XImats[163] = static_cast<T>(0);
        h_XImats[164] = static_cast<T>(0);
        h_XImats[165] = static_cast<T>(0);
        h_XImats[166] = static_cast<T>(0);
        h_XImats[167] = static_cast<T>(0);
        h_XImats[168] = static_cast<T>(0);
        h_XImats[169] = static_cast<T>(0);
        h_XImats[170] = static_cast<T>(0);
        h_XImats[171] = static_cast<T>(0);
        h_XImats[172] = static_cast<T>(0);
        h_XImats[173] = static_cast<T>(1.00000000000000);
        h_XImats[174] = static_cast<T>(0);
        h_XImats[175] = static_cast<T>(0);
        h_XImats[176] = static_cast<T>(0);
        h_XImats[177] = static_cast<T>(0);
        h_XImats[178] = static_cast<T>(0);
        h_XImats[179] = static_cast<T>(0);
        // X[5]
        h_XImats[180] = static_cast<T>(0);
        h_XImats[181] = static_cast<T>(0);
        h_XImats[182] = static_cast<T>(-1.60982338570648e-15);
        h_XImats[183] = static_cast<T>(0);
        h_XImats[184] = static_cast<T>(0);
        h_XImats[185] = static_cast<T>(-0.215500000000000);
        h_XImats[186] = static_cast<T>(0);
        h_XImats[187] = static_cast<T>(0);
        h_XImats[188] = static_cast<T>(1.00000000000000);
        h_XImats[189] = static_cast<T>(0);
        h_XImats[190] = static_cast<T>(0);
        h_XImats[191] = static_cast<T>(0);
        h_XImats[192] = static_cast<T>(0);
        h_XImats[193] = static_cast<T>(0);
        h_XImats[194] = static_cast<T>(0);
        h_XImats[195] = static_cast<T>(0);
        h_XImats[196] = static_cast<T>(0);
        h_XImats[197] = static_cast<T>(0);
        h_XImats[198] = static_cast<T>(0);
        h_XImats[199] = static_cast<T>(0);
        h_XImats[200] = static_cast<T>(0);
        h_XImats[201] = static_cast<T>(0);
        h_XImats[202] = static_cast<T>(0);
        h_XImats[203] = static_cast<T>(-1.60982338570648e-15);
        h_XImats[204] = static_cast<T>(0);
        h_XImats[205] = static_cast<T>(0);
        h_XImats[206] = static_cast<T>(0);
        h_XImats[207] = static_cast<T>(0);
        h_XImats[208] = static_cast<T>(0);
        h_XImats[209] = static_cast<T>(1.00000000000000);
        h_XImats[210] = static_cast<T>(0);
        h_XImats[211] = static_cast<T>(0);
        h_XImats[212] = static_cast<T>(0);
        h_XImats[213] = static_cast<T>(0);
        h_XImats[214] = static_cast<T>(0);
        h_XImats[215] = static_cast<T>(0);
        // X[6]
        h_XImats[216] = static_cast<T>(0);
        h_XImats[217] = static_cast<T>(0);
        h_XImats[218] = static_cast<T>(1.00000000000000);
        h_XImats[219] = static_cast<T>(0);
        h_XImats[220] = static_cast<T>(0);
        h_XImats[221] = static_cast<T>(0);
        h_XImats[222] = static_cast<T>(0);
        h_XImats[223] = static_cast<T>(0);
        h_XImats[224] = static_cast<T>(0);
        h_XImats[225] = static_cast<T>(0);
        h_XImats[226] = static_cast<T>(0);
        h_XImats[227] = static_cast<T>(0.0607000000000000);
        h_XImats[228] = static_cast<T>(0);
        h_XImats[229] = static_cast<T>(0);
        h_XImats[230] = static_cast<T>(0);
        h_XImats[231] = static_cast<T>(0);
        h_XImats[232] = static_cast<T>(0);
        h_XImats[233] = static_cast<T>(0);
        h_XImats[234] = static_cast<T>(0);
        h_XImats[235] = static_cast<T>(0);
        h_XImats[236] = static_cast<T>(0);
        h_XImats[237] = static_cast<T>(0);
        h_XImats[238] = static_cast<T>(0);
        h_XImats[239] = static_cast<T>(1.00000000000000);
        h_XImats[240] = static_cast<T>(0);
        h_XImats[241] = static_cast<T>(0);
        h_XImats[242] = static_cast<T>(0);
        h_XImats[243] = static_cast<T>(0);
        h_XImats[244] = static_cast<T>(0);
        h_XImats[245] = static_cast<T>(0);
        h_XImats[246] = static_cast<T>(0);
        h_XImats[247] = static_cast<T>(0);
        h_XImats[248] = static_cast<T>(0);
        h_XImats[249] = static_cast<T>(0);
        h_XImats[250] = static_cast<T>(0);
        h_XImats[251] = static_cast<T>(0);
        // I[0]
        h_XImats[252] = static_cast<T>(0.12112799999999999);
        h_XImats[253] = static_cast<T>(0.0);
        h_XImats[254] = static_cast<T>(0.0);
        h_XImats[255] = static_cast<T>(0.0);
        h_XImats[256] = static_cast<T>(-0.6911999999999999);
        h_XImats[257] = static_cast<T>(-0.17279999999999998);
        h_XImats[258] = static_cast<T>(0.0);
        h_XImats[259] = static_cast<T>(0.11624399999999999);
        h_XImats[260] = static_cast<T>(0.020735999999999997);
        h_XImats[261] = static_cast<T>(0.6911999999999999);
        h_XImats[262] = static_cast<T>(0.0);
        h_XImats[263] = static_cast<T>(0.0);
        h_XImats[264] = static_cast<T>(0.0);
        h_XImats[265] = static_cast<T>(0.020735999999999997);
        h_XImats[266] = static_cast<T>(0.017484);
        h_XImats[267] = static_cast<T>(0.17279999999999998);
        h_XImats[268] = static_cast<T>(0.0);
        h_XImats[269] = static_cast<T>(0.0);
        h_XImats[270] = static_cast<T>(0.0);
        h_XImats[271] = static_cast<T>(0.6911999999999999);
        h_XImats[272] = static_cast<T>(0.17279999999999998);
        h_XImats[273] = static_cast<T>(5.76);
        h_XImats[274] = static_cast<T>(0.0);
        h_XImats[275] = static_cast<T>(0.0);
        h_XImats[276] = static_cast<T>(-0.6911999999999999);
        h_XImats[277] = static_cast<T>(0.0);
        h_XImats[278] = static_cast<T>(0.0);
        h_XImats[279] = static_cast<T>(0.0);
        h_XImats[280] = static_cast<T>(5.76);
        h_XImats[281] = static_cast<T>(0.0);
        h_XImats[282] = static_cast<T>(-0.17279999999999998);
        h_XImats[283] = static_cast<T>(0.0);
        h_XImats[284] = static_cast<T>(0.0);
        h_XImats[285] = static_cast<T>(0.0);
        h_XImats[286] = static_cast<T>(0.0);
        h_XImats[287] = static_cast<T>(5.76);
        // I[1]
        h_XImats[288] = static_cast<T>(0.04160197057286055);
        h_XImats[289] = static_cast<T>(0.00011239454317625987);
        h_XImats[290] = static_cast<T>(-0.01573530024735758);
        h_XImats[291] = static_cast<T>(0.0);
        h_XImats[292] = static_cast<T>(-0.266699989017675);
        h_XImats[293] = static_cast<T>(-0.0019049922930431);
        h_XImats[294] = static_cast<T>(0.00011239454317625987);
        h_XImats[295] = static_cast<T>(0.06380575000459363);
        h_XImats[296] = static_cast<T>(8.001033406865006e-05);
        h_XImats[297] = static_cast<T>(0.266699989017675);
        h_XImats[298] = static_cast<T>(0.0);
        h_XImats[299] = static_cast<T>(-0.374650007856855);
        h_XImats[300] = static_cast<T>(-0.01573530024735758);
        h_XImats[301] = static_cast<T>(8.001033406865006e-05);
        h_XImats[302] = static_cast<T>(0.03310492242248474);
        h_XImats[303] = static_cast<T>(0.0019049922930431);
        h_XImats[304] = static_cast<T>(0.374650007856855);
        h_XImats[305] = static_cast<T>(0.0);
        h_XImats[306] = static_cast<T>(0.0);
        h_XImats[307] = static_cast<T>(0.266699989017675);
        h_XImats[308] = static_cast<T>(0.0019049922930431);
        h_XImats[309] = static_cast<T>(6.35);
        h_XImats[310] = static_cast<T>(0.0);
        h_XImats[311] = static_cast<T>(0.0);
        h_XImats[312] = static_cast<T>(-0.266699989017675);
        h_XImats[313] = static_cast<T>(0.0);
        h_XImats[314] = static_cast<T>(0.374650007856855);
        h_XImats[315] = static_cast<T>(0.0);
        h_XImats[316] = static_cast<T>(6.35);
        h_XImats[317] = static_cast<T>(0.0);
        h_XImats[318] = static_cast<T>(-0.0019049922930431);
        h_XImats[319] = static_cast<T>(-0.374650007856855);
        h_XImats[320] = static_cast<T>(0.0);
        h_XImats[321] = static_cast<T>(0.0);
        h_XImats[322] = static_cast<T>(0.0);
        h_XImats[323] = static_cast<T>(6.35);
        // I[2]
        h_XImats[324] = static_cast<T>(0.08729999808545157);
        h_XImats[325] = static_cast<T>(-5.260676579451288e-10);
        h_XImats[326] = static_cast<T>(-2.0418312232572323e-09);
        h_XImats[327] = static_cast<T>(0.0);
        h_XImats[328] = static_cast<T>(-0.4550000028455);
        h_XImats[329] = static_cast<T>(0.10499995576035);
        h_XImats[330] = static_cast<T>(-5.260676579451287e-10);
        h_XImats[331] = static_cast<T>(0.08295000073983008);
        h_XImats[332] = static_cast<T>(-0.013649993895035295);
        h_XImats[333] = static_cast<T>(0.4550000028455);
        h_XImats[334] = static_cast<T>(0.0);
        h_XImats[335] = static_cast<T>(-1.7143151552345002e-08);
        h_XImats[336] = static_cast<T>(-2.0418312232572323e-09);
        h_XImats[337] = static_cast<T>(-0.013649993895035297);
        h_XImats[338] = static_cast<T>(0.010749997345621643);
        h_XImats[339] = static_cast<T>(-0.10499995576035);
        h_XImats[340] = static_cast<T>(1.7143151552345002e-08);
        h_XImats[341] = static_cast<T>(0.0);
        h_XImats[342] = static_cast<T>(0.0);
        h_XImats[343] = static_cast<T>(0.4550000028455);
        h_XImats[344] = static_cast<T>(-0.10499995576035);
        h_XImats[345] = static_cast<T>(3.5);
        h_XImats[346] = static_cast<T>(0.0);
        h_XImats[347] = static_cast<T>(0.0);
        h_XImats[348] = static_cast<T>(-0.4550000028455);
        h_XImats[349] = static_cast<T>(0.0);
        h_XImats[350] = static_cast<T>(1.7143151552345002e-08);
        h_XImats[351] = static_cast<T>(0.0);
        h_XImats[352] = static_cast<T>(3.5);
        h_XImats[353] = static_cast<T>(0.0);
        h_XImats[354] = static_cast<T>(0.10499995576035);
        h_XImats[355] = static_cast<T>(-1.7143151552345002e-08);
        h_XImats[356] = static_cast<T>(0.0);
        h_XImats[357] = static_cast<T>(0.0);
        h_XImats[358] = static_cast<T>(0.0);
        h_XImats[359] = static_cast<T>(3.5);
        // I[3]
        h_XImats[360] = static_cast<T>(0.03675750325128786);
        h_XImats[361] = static_cast<T>(-6.220314824991959e-10);
        h_XImats[362] = static_cast<T>(-2.453491357183656e-10);
        h_XImats[363] = static_cast<T>(0.0);
        h_XImats[364] = static_cast<T>(-0.119000048111);
        h_XImats[365] = static_cast<T>(0.23449999984880002);
        h_XImats[366] = static_cast<T>(-6.220314824991959e-10);
        h_XImats[367] = static_cast<T>(0.020446003271548687);
        h_XImats[368] = static_cast<T>(-0.007973003406400392);
        h_XImats[369] = static_cast<T>(0.119000048111);
        h_XImats[370] = static_cast<T>(0.0);
        h_XImats[371] = static_cast<T>(-9.380180836949994e-09);
        h_XImats[372] = static_cast<T>(-2.4534913571836563e-10);
        h_XImats[373] = static_cast<T>(-0.007973003406400392);
        h_XImats[374] = static_cast<T>(0.02171149997973923);
        h_XImats[375] = static_cast<T>(-0.23449999984880002);
        h_XImats[376] = static_cast<T>(9.380180836949994e-09);
        h_XImats[377] = static_cast<T>(0.0);
        h_XImats[378] = static_cast<T>(0.0);
        h_XImats[379] = static_cast<T>(0.119000048111);
        h_XImats[380] = static_cast<T>(-0.23449999984880002);
        h_XImats[381] = static_cast<T>(3.5);
        h_XImats[382] = static_cast<T>(0.0);
        h_XImats[383] = static_cast<T>(0.0);
        h_XImats[384] = static_cast<T>(-0.119000048111);
        h_XImats[385] = static_cast<T>(0.0);
        h_XImats[386] = static_cast<T>(9.380180836949994e-09);
        h_XImats[387] = static_cast<T>(0.0);
        h_XImats[388] = static_cast<T>(3.5);
        h_XImats[389] = static_cast<T>(0.0);
        h_XImats[390] = static_cast<T>(0.23449999984880002);
        h_XImats[391] = static_cast<T>(-9.380180836949994e-09);
        h_XImats[392] = static_cast<T>(0.0);
        h_XImats[393] = static_cast<T>(0.0);
        h_XImats[394] = static_cast<T>(0.0);
        h_XImats[395] = static_cast<T>(3.5);
        // I[4]
        h_XImats[396] = static_cast<T>(0.031759501206084464);
        h_XImats[397] = static_cast<T>(-7.349665418713475e-06);
        h_XImats[398] = static_cast<T>(2.6598755117970484e-05);
        h_XImats[399] = static_cast<T>(0.0);
        h_XImats[400] = static_cast<T>(-0.2659999956621);
        h_XImats[401] = static_cast<T>(-0.07350004441535);
        h_XImats[402] = static_cast<T>(-7.349665418713476e-06);
        h_XImats[403] = static_cast<T>(0.02891603433732768);
        h_XImats[404] = static_cast<T>(0.005586003400945494);
        h_XImats[405] = static_cast<T>(0.2659999956621);
        h_XImats[406] = static_cast<T>(0.0);
        h_XImats[407] = static_cast<T>(0.0003499834419956);
        h_XImats[408] = static_cast<T>(2.6598755117970484e-05);
        h_XImats[409] = static_cast<T>(0.005586003400945493);
        h_XImats[410] = static_cast<T>(0.006033536862133741);
        h_XImats[411] = static_cast<T>(0.07350004441535);
        h_XImats[412] = static_cast<T>(-0.0003499834419956);
        h_XImats[413] = static_cast<T>(0.0);
        h_XImats[414] = static_cast<T>(0.0);
        h_XImats[415] = static_cast<T>(0.2659999956621);
        h_XImats[416] = static_cast<T>(0.07350004441535);
        h_XImats[417] = static_cast<T>(3.5);
        h_XImats[418] = static_cast<T>(0.0);
        h_XImats[419] = static_cast<T>(0.0);
        h_XImats[420] = static_cast<T>(-0.2659999956621);
        h_XImats[421] = static_cast<T>(0.0);
        h_XImats[422] = static_cast<T>(-0.0003499834419956);
        h_XImats[423] = static_cast<T>(0.0);
        h_XImats[424] = static_cast<T>(3.5);
        h_XImats[425] = static_cast<T>(0.0);
        h_XImats[426] = static_cast<T>(-0.07350004441535);
        h_XImats[427] = static_cast<T>(0.0003499834419956);
        h_XImats[428] = static_cast<T>(0.0);
        h_XImats[429] = static_cast<T>(0.0);
        h_XImats[430] = static_cast<T>(0.0);
        h_XImats[431] = static_cast<T>(3.5);
        // I[5]
        h_XImats[432] = static_cast<T>(0.011419774351150058);
        h_XImats[433] = static_cast<T>(0.0);
        h_XImats[434] = static_cast<T>(-6.598851780613524e-05);
        h_XImats[435] = static_cast<T>(0.0);
        h_XImats[436] = static_cast<T>(-0.10997997014034);
        h_XImats[437] = static_cast<T>(4.768652207304001e-09);
        h_XImats[438] = static_cast<T>(0.0);
        h_XImats[439] = static_cast<T>(0.011620422360287286);
        h_XImats[440] = static_cast<T>(-2.467813722148336e-10);
        h_XImats[441] = static_cast<T>(0.10997997014034);
        h_XImats[442] = static_cast<T>(0.0);
        h_XImats[443] = static_cast<T>(-0.001080007614342);
        h_XImats[444] = static_cast<T>(-6.598851780613524e-05);
        h_XImats[445] = static_cast<T>(-2.467813722148336e-10);
        h_XImats[446] = static_cast<T>(0.0036006480091372553);
        h_XImats[447] = static_cast<T>(-4.768652207304001e-09);
        h_XImats[448] = static_cast<T>(0.001080007614342);
        h_XImats[449] = static_cast<T>(0.0);
        h_XImats[450] = static_cast<T>(0.0);
        h_XImats[451] = static_cast<T>(0.10997997014034);
        h_XImats[452] = static_cast<T>(-4.768652207304001e-09);
        h_XImats[453] = static_cast<T>(1.8);
        h_XImats[454] = static_cast<T>(0.0);
        h_XImats[455] = static_cast<T>(0.0);
        h_XImats[456] = static_cast<T>(-0.10997997014034);
        h_XImats[457] = static_cast<T>(0.0);
        h_XImats[458] = static_cast<T>(0.001080007614342);
        h_XImats[459] = static_cast<T>(0.0);
        h_XImats[460] = static_cast<T>(1.8);
        h_XImats[461] = static_cast<T>(0.0);
        h_XImats[462] = static_cast<T>(4.768652207304001e-09);
        h_XImats[463] = static_cast<T>(-0.001080007614342);
        h_XImats[464] = static_cast<T>(0.0);
        h_XImats[465] = static_cast<T>(0.0);
        h_XImats[466] = static_cast<T>(0.0);
        h_XImats[467] = static_cast<T>(1.8);
        // I[6]
        h_XImats[468] = static_cast<T>(0.0014800002018213413);
        h_XImats[469] = static_cast<T>(0.0);
        h_XImats[470] = static_cast<T>(-2.7657300069272947e-10);
        h_XImats[471] = static_cast<T>(0.0);
        h_XImats[472] = static_cast<T>(-0.02400000504552);
        h_XImats[473] = static_cast<T>(-2.498451792768e-08);
        h_XImats[474] = static_cast<T>(0.0);
        h_XImats[475] = static_cast<T>(0.0014800002018209805);
        h_XImats[476] = static_cast<T>(4.99690463603504e-10);
        h_XImats[477] = static_cast<T>(0.02400000504552);
        h_XImats[478] = static_cast<T>(0.0);
        h_XImats[479] = static_cast<T>(-1.3828647127439989e-08);
        h_XImats[480] = static_cast<T>(-2.765730006927295e-10);
        h_XImats[481] = static_cast<T>(4.996904636035041e-10);
        h_XImats[482] = static_cast<T>(0.0010000000000006796);
        h_XImats[483] = static_cast<T>(2.498451792768e-08);
        h_XImats[484] = static_cast<T>(1.3828647127439989e-08);
        h_XImats[485] = static_cast<T>(0.0);
        h_XImats[486] = static_cast<T>(0.0);
        h_XImats[487] = static_cast<T>(0.02400000504552);
        h_XImats[488] = static_cast<T>(2.498451792768e-08);
        h_XImats[489] = static_cast<T>(1.2);
        h_XImats[490] = static_cast<T>(0.0);
        h_XImats[491] = static_cast<T>(0.0);
        h_XImats[492] = static_cast<T>(-0.02400000504552);
        h_XImats[493] = static_cast<T>(0.0);
        h_XImats[494] = static_cast<T>(1.3828647127439989e-08);
        h_XImats[495] = static_cast<T>(0.0);
        h_XImats[496] = static_cast<T>(1.2);
        h_XImats[497] = static_cast<T>(0.0);
        h_XImats[498] = static_cast<T>(-2.498451792768e-08);
        h_XImats[499] = static_cast<T>(-1.3828647127439989e-08);
        h_XImats[500] = static_cast<T>(0.0);
        h_XImats[501] = static_cast<T>(0.0);
        h_XImats[502] = static_cast<T>(0.0);
        h_XImats[503] = static_cast<T>(1.2);
        // Xhom[0]
        h_XImats[504] = static_cast<T>(0);
        h_XImats[505] = static_cast<T>(0);
        h_XImats[506] = static_cast<T>(0);
        h_XImats[507] = static_cast<T>(0);
        h_XImats[508] = static_cast<T>(0);
        h_XImats[509] = static_cast<T>(0);
        h_XImats[510] = static_cast<T>(0);
        h_XImats[511] = static_cast<T>(0);
        h_XImats[512] = static_cast<T>(0);
        h_XImats[513] = static_cast<T>(0);
        h_XImats[514] = static_cast<T>(1.00000000000000);
        h_XImats[515] = static_cast<T>(0);
        h_XImats[516] = static_cast<T>(0);
        h_XImats[517] = static_cast<T>(0);
        h_XImats[518] = static_cast<T>(0.157500000000000);
        h_XImats[519] = static_cast<T>(1.00000000000000);
        // Xhom[1]
        h_XImats[520] = static_cast<T>(0);
        h_XImats[521] = static_cast<T>(0);
        h_XImats[522] = static_cast<T>(0);
        h_XImats[523] = static_cast<T>(0);
        h_XImats[524] = static_cast<T>(0);
        h_XImats[525] = static_cast<T>(0);
        h_XImats[526] = static_cast<T>(0);
        h_XImats[527] = static_cast<T>(0);
        h_XImats[528] = static_cast<T>(-1.60982338570648e-15);
        h_XImats[529] = static_cast<T>(1.00000000000000);
        h_XImats[530] = static_cast<T>(0);
        h_XImats[531] = static_cast<T>(0);
        h_XImats[532] = static_cast<T>(0);
        h_XImats[533] = static_cast<T>(0);
        h_XImats[534] = static_cast<T>(0.202500000000000);
        h_XImats[535] = static_cast<T>(1.00000000000000);
        // Xhom[2]
        h_XImats[536] = static_cast<T>(0);
        h_XImats[537] = static_cast<T>(0);
        h_XImats[538] = static_cast<T>(0);
        h_XImats[539] = static_cast<T>(0);
        h_XImats[540] = static_cast<T>(0);
        h_XImats[541] = static_cast<T>(0);
        h_XImats[542] = static_cast<T>(0);
        h_XImats[543] = static_cast<T>(0);
        h_XImats[544] = static_cast<T>(1.00000000000000);
        h_XImats[545] = static_cast<T>(0);
        h_XImats[546] = static_cast<T>(0);
        h_XImats[547] = static_cast<T>(0);
        h_XImats[548] = static_cast<T>(0.204500000000000);
        h_XImats[549] = static_cast<T>(0);
        h_XImats[550] = static_cast<T>(0);
        h_XImats[551] = static_cast<T>(1.00000000000000);
        // Xhom[3]
        h_XImats[552] = static_cast<T>(0);
        h_XImats[553] = static_cast<T>(0);
        h_XImats[554] = static_cast<T>(0);
        h_XImats[555] = static_cast<T>(0);
        h_XImats[556] = static_cast<T>(0);
        h_XImats[557] = static_cast<T>(0);
        h_XImats[558] = static_cast<T>(0);
        h_XImats[559] = static_cast<T>(0);
        h_XImats[560] = static_cast<T>(0);
        h_XImats[561] = static_cast<T>(-1.00000000000000);
        h_XImats[562] = static_cast<T>(0);
        h_XImats[563] = static_cast<T>(0);
        h_XImats[564] = static_cast<T>(0);
        h_XImats[565] = static_cast<T>(0);
        h_XImats[566] = static_cast<T>(0.215500000000000);
        h_XImats[567] = static_cast<T>(1.00000000000000);
        // Xhom[4]
        h_XImats[568] = static_cast<T>(0);
        h_XImats[569] = static_cast<T>(0);
        h_XImats[570] = static_cast<T>(0);
        h_XImats[571] = static_cast<T>(0);
        h_XImats[572] = static_cast<T>(0);
        h_XImats[573] = static_cast<T>(0);
        h_XImats[574] = static_cast<T>(0);
        h_XImats[575] = static_cast<T>(0);
        h_XImats[576] = static_cast<T>(0);
        h_XImats[577] = static_cast<T>(1.00000000000000);
        h_XImats[578] = static_cast<T>(0);
        h_XImats[579] = static_cast<T>(0);
        h_XImats[580] = static_cast<T>(0);
        h_XImats[581] = static_cast<T>(0.184500000000000);
        h_XImats[582] = static_cast<T>(0);
        h_XImats[583] = static_cast<T>(1.00000000000000);
        // Xhom[5]
        h_XImats[584] = static_cast<T>(0);
        h_XImats[585] = static_cast<T>(0);
        h_XImats[586] = static_cast<T>(0);
        h_XImats[587] = static_cast<T>(0);
        h_XImats[588] = static_cast<T>(0);
        h_XImats[589] = static_cast<T>(0);
        h_XImats[590] = static_cast<T>(0);
        h_XImats[591] = static_cast<T>(0);
        h_XImats[592] = static_cast<T>(-1.60982338570648e-15);
        h_XImats[593] = static_cast<T>(1.00000000000000);
        h_XImats[594] = static_cast<T>(0);
        h_XImats[595] = static_cast<T>(0);
        h_XImats[596] = static_cast<T>(0);
        h_XImats[597] = static_cast<T>(-0.0607000000000000);
        h_XImats[598] = static_cast<T>(0.215500000000000);
        h_XImats[599] = static_cast<T>(1.00000000000000);
        // Xhom[6]
        h_XImats[600] = static_cast<T>(0);
        h_XImats[601] = static_cast<T>(0);
        h_XImats[602] = static_cast<T>(0);
        h_XImats[603] = static_cast<T>(0);
        h_XImats[604] = static_cast<T>(0);
        h_XImats[605] = static_cast<T>(0);
        h_XImats[606] = static_cast<T>(0);
        h_XImats[607] = static_cast<T>(0);
        h_XImats[608] = static_cast<T>(1.00000000000000);
        h_XImats[609] = static_cast<T>(0);
        h_XImats[610] = static_cast<T>(0);
        h_XImats[611] = static_cast<T>(0);
        h_XImats[612] = static_cast<T>(0.0810000000000000);
        h_XImats[613] = static_cast<T>(0);
        h_XImats[614] = static_cast<T>(0.0607000000000000);
        h_XImats[615] = static_cast<T>(1.00000000000000);
        // Xhom[7]
        h_XImats[616] = static_cast<T>(1.0);
        h_XImats[617] = static_cast<T>(0.0);
        h_XImats[618] = static_cast<T>(0.0);
        h_XImats[619] = static_cast<T>(0.0);
        h_XImats[620] = static_cast<T>(0.0);
        h_XImats[621] = static_cast<T>(1.0);
        h_XImats[622] = static_cast<T>(0.0);
        h_XImats[623] = static_cast<T>(0.0);
        h_XImats[624] = static_cast<T>(0.0);
        h_XImats[625] = static_cast<T>(0.0);
        h_XImats[626] = static_cast<T>(1.0);
        h_XImats[627] = static_cast<T>(0.0);
        h_XImats[628] = static_cast<T>(0.0);
        h_XImats[629] = static_cast<T>(0.0);
        h_XImats[630] = static_cast<T>(0.04);
        h_XImats[631] = static_cast<T>(1.0);
        // Xhom[8]
        h_XImats[632] = static_cast<T>(1.0);
        h_XImats[633] = static_cast<T>(0.0);
        h_XImats[634] = static_cast<T>(0.0);
        h_XImats[635] = static_cast<T>(0.0);
        h_XImats[636] = static_cast<T>(0.0);
        h_XImats[637] = static_cast<T>(1.0);
        h_XImats[638] = static_cast<T>(0.0);
        h_XImats[639] = static_cast<T>(0.0);
        h_XImats[640] = static_cast<T>(0.0);
        h_XImats[641] = static_cast<T>(0.0);
        h_XImats[642] = static_cast<T>(1.0);
        h_XImats[643] = static_cast<T>(0.0);
        h_XImats[644] = static_cast<T>(0.0);
        h_XImats[645] = static_cast<T>(0.0);
        h_XImats[646] = static_cast<T>(0.0);
        h_XImats[647] = static_cast<T>(1.0);
        // dXhom[0]
        h_XImats[648] = static_cast<T>(0);
        h_XImats[649] = static_cast<T>(0);
        h_XImats[650] = static_cast<T>(0);
        h_XImats[651] = static_cast<T>(0);
        h_XImats[652] = static_cast<T>(0);
        h_XImats[653] = static_cast<T>(0);
        h_XImats[654] = static_cast<T>(0);
        h_XImats[655] = static_cast<T>(0);
        h_XImats[656] = static_cast<T>(0);
        h_XImats[657] = static_cast<T>(0);
        h_XImats[658] = static_cast<T>(0);
        h_XImats[659] = static_cast<T>(0);
        h_XImats[660] = static_cast<T>(0);
        h_XImats[661] = static_cast<T>(0);
        h_XImats[662] = static_cast<T>(0);
        h_XImats[663] = static_cast<T>(0);
        // dXhom[1]
        h_XImats[664] = static_cast<T>(0);
        h_XImats[665] = static_cast<T>(0);
        h_XImats[666] = static_cast<T>(0);
        h_XImats[667] = static_cast<T>(0);
        h_XImats[668] = static_cast<T>(0);
        h_XImats[669] = static_cast<T>(0);
        h_XImats[670] = static_cast<T>(0);
        h_XImats[671] = static_cast<T>(0);
        h_XImats[672] = static_cast<T>(0);
        h_XImats[673] = static_cast<T>(0);
        h_XImats[674] = static_cast<T>(0);
        h_XImats[675] = static_cast<T>(0);
        h_XImats[676] = static_cast<T>(0);
        h_XImats[677] = static_cast<T>(0);
        h_XImats[678] = static_cast<T>(0);
        h_XImats[679] = static_cast<T>(0);
        // dXhom[2]
        h_XImats[680] = static_cast<T>(0);
        h_XImats[681] = static_cast<T>(0);
        h_XImats[682] = static_cast<T>(0);
        h_XImats[683] = static_cast<T>(0);
        h_XImats[684] = static_cast<T>(0);
        h_XImats[685] = static_cast<T>(0);
        h_XImats[686] = static_cast<T>(0);
        h_XImats[687] = static_cast<T>(0);
        h_XImats[688] = static_cast<T>(0);
        h_XImats[689] = static_cast<T>(0);
        h_XImats[690] = static_cast<T>(0);
        h_XImats[691] = static_cast<T>(0);
        h_XImats[692] = static_cast<T>(0);
        h_XImats[693] = static_cast<T>(0);
        h_XImats[694] = static_cast<T>(0);
        h_XImats[695] = static_cast<T>(0);
        // dXhom[3]
        h_XImats[696] = static_cast<T>(0);
        h_XImats[697] = static_cast<T>(0);
        h_XImats[698] = static_cast<T>(0);
        h_XImats[699] = static_cast<T>(0);
        h_XImats[700] = static_cast<T>(0);
        h_XImats[701] = static_cast<T>(0);
        h_XImats[702] = static_cast<T>(0);
        h_XImats[703] = static_cast<T>(0);
        h_XImats[704] = static_cast<T>(0);
        h_XImats[705] = static_cast<T>(0);
        h_XImats[706] = static_cast<T>(0);
        h_XImats[707] = static_cast<T>(0);
        h_XImats[708] = static_cast<T>(0);
        h_XImats[709] = static_cast<T>(0);
        h_XImats[710] = static_cast<T>(0);
        h_XImats[711] = static_cast<T>(0);
        // dXhom[4]
        h_XImats[712] = static_cast<T>(0);
        h_XImats[713] = static_cast<T>(0);
        h_XImats[714] = static_cast<T>(0);
        h_XImats[715] = static_cast<T>(0);
        h_XImats[716] = static_cast<T>(0);
        h_XImats[717] = static_cast<T>(0);
        h_XImats[718] = static_cast<T>(0);
        h_XImats[719] = static_cast<T>(0);
        h_XImats[720] = static_cast<T>(0);
        h_XImats[721] = static_cast<T>(0);
        h_XImats[722] = static_cast<T>(0);
        h_XImats[723] = static_cast<T>(0);
        h_XImats[724] = static_cast<T>(0);
        h_XImats[725] = static_cast<T>(0);
        h_XImats[726] = static_cast<T>(0);
        h_XImats[727] = static_cast<T>(0);
        // dXhom[5]
        h_XImats[728] = static_cast<T>(0);
        h_XImats[729] = static_cast<T>(0);
        h_XImats[730] = static_cast<T>(0);
        h_XImats[731] = static_cast<T>(0);
        h_XImats[732] = static_cast<T>(0);
        h_XImats[733] = static_cast<T>(0);
        h_XImats[734] = static_cast<T>(0);
        h_XImats[735] = static_cast<T>(0);
        h_XImats[736] = static_cast<T>(0);
        h_XImats[737] = static_cast<T>(0);
        h_XImats[738] = static_cast<T>(0);
        h_XImats[739] = static_cast<T>(0);
        h_XImats[740] = static_cast<T>(0);
        h_XImats[741] = static_cast<T>(0);
        h_XImats[742] = static_cast<T>(0);
        h_XImats[743] = static_cast<T>(0);
        // dXhom[6]
        h_XImats[744] = static_cast<T>(0);
        h_XImats[745] = static_cast<T>(0);
        h_XImats[746] = static_cast<T>(0);
        h_XImats[747] = static_cast<T>(0);
        h_XImats[748] = static_cast<T>(0);
        h_XImats[749] = static_cast<T>(0);
        h_XImats[750] = static_cast<T>(0);
        h_XImats[751] = static_cast<T>(0);
        h_XImats[752] = static_cast<T>(0);
        h_XImats[753] = static_cast<T>(0);
        h_XImats[754] = static_cast<T>(0);
        h_XImats[755] = static_cast<T>(0);
        h_XImats[756] = static_cast<T>(0);
        h_XImats[757] = static_cast<T>(0);
        h_XImats[758] = static_cast<T>(0);
        h_XImats[759] = static_cast<T>(0);
        T *d_XImats = nullptr;
        cudaError_t _e = GRID_CUDA_CALL(cudaMalloc((void**)&d_XImats,872*sizeof(T)));
        if (_e != cudaSuccess) { free(h_XImats); return grid_fail(failed_op, "cudaMalloc(d_XImats)", _e); }
        _e = GRID_CUDA_CALL(cudaMemcpy(d_XImats,h_XImats,872*sizeof(T),cudaMemcpyHostToDevice));
        free(h_XImats);
        if (_e != cudaSuccess) { grid_cleanup_free(d_XImats, "cudaFree(d_XImats)", nullptr, nullptr); return grid_fail(failed_op, "cudaMemcpy(d_XImats)", _e); }
        *out = d_XImats;
        return cudaSuccess;
    }

    template <typename T>
    __host__
    T* init_XImats() {
        T *d = nullptr; const char *op = nullptr;
        cudaError_t e = init_XImats_checked<T>(&d, &op);  // sequenced BEFORE reading op
        grid_legacy_check(e, op, __FILE__, __LINE__);
        return d;
    }

    /**
     * Releases the nested device arrays a host-side robotModel<T> owns (best effort, reverse order; null members skipped)
     *
     * @param h_robotModel is the host copy of the struct; members are nulled as they are released
     * @return the first cudaFree error (cudaSuccess if none); the failing member is named in *cleanup_op
     */
    template <typename T>
    __host__
    cudaError_t release_robotModel_members(robotModel<T> &h_robotModel, const char **cleanup_op = nullptr) {
        cudaError_t first = cudaSuccess;
        grid_cleanup_free(h_robotModel.d_topology_helpers, "cudaFree(d_topology_helpers)", &first, cleanup_op); h_robotModel.d_topology_helpers = nullptr;
        grid_cleanup_free(h_robotModel.d_XImats, "cudaFree(d_XImats)", &first, cleanup_op); h_robotModel.d_XImats = nullptr;
        return first;
    }

    /**
     * Library-safe initialization of the robotModel helpers in GPU memory: every owned pointer is null before the first fallible call, construction stops at the first failure, everything acquired by this attempt is released, and *out is published only on complete success (never exit/abort/cudaDeviceReset)
     *
     * Notes:
     *   The allocating device must be current; the model is bound to it (see free_robotModel_checked)
     *
     * @param out receives the device-resident struct pointer (nullptr on failure)
     * @param failed_op (optional) receives a static string naming the failed operation
     * @return cudaSuccess, or the first CUDA error (cudaErrorMemoryAllocation also stands for a failed host allocation)
     */
    template <typename T>
    __host__
    cudaError_t init_robotModel_checked(robotModel<T> **out, const char **failed_op = nullptr) {
        *out = nullptr;
        robotModel<T> h_robotModel = {};  // every owned pointer null before any fallible work
        cudaError_t e = cudaSuccess;
        e = init_XImats_checked<T>(&h_robotModel.d_XImats, failed_op);
        if (e != cudaSuccess) { release_robotModel_members<T>(h_robotModel); return e; }
        e = init_topology_helpers_checked<T>(&h_robotModel.d_topology_helpers, failed_op);
        if (e != cudaSuccess) { release_robotModel_members<T>(h_robotModel); return e; }
        robotModel<T> *d_robotModel = nullptr;
        e = GRID_CUDA_CALL(cudaMalloc((void**)&d_robotModel,sizeof(robotModel<T>)));
        if (e != cudaSuccess) { release_robotModel_members<T>(h_robotModel); return grid_fail(failed_op, "cudaMalloc(d_robotModel)", e); }
        e = GRID_CUDA_CALL(cudaMemcpy(d_robotModel,&h_robotModel,sizeof(robotModel<T>),cudaMemcpyHostToDevice));
        if (e != cudaSuccess) { grid_cleanup_free(d_robotModel, "cudaFree(d_robotModel)", nullptr, nullptr); release_robotModel_members<T>(h_robotModel); return grid_fail(failed_op, "cudaMemcpy(d_robotModel)", e); }
        *out = d_robotModel;
        return cudaSuccess;
    }

    /**
     * Initializes the robotModel helpers in GPU memory (legacy policy: exit on failure, or sticky first error + nullptr under GRID_GPUERRCHK_NO_EXIT; prefer init_robotModel_checked in library code)
     *
     * @return A pointer to the robotModel struct
     */
    template <typename T>
    __host__
    robotModel<T>* init_robotModel() {
        robotModel<T> *d_robotModel = nullptr; const char *op = nullptr;
        cudaError_t e = init_robotModel_checked<T>(&d_robotModel, &op);  // sequenced BEFORE reading op
        grid_legacy_check(e, op, __FILE__, __LINE__);
        return d_robotModel;
    }

    /**
     * Library-safe destruction of a robotModel allocated by init_robotModel[_checked]: frees the NESTED device arrays (d_XImats / d_topology_helpers [+ any flag-gated runtime parameter tables]) AND the struct itself, without exit/abort/cudaDeviceReset. nullptr is a no-op (cudaSuccess). The struct is copied back to recover the nested pointers; if THAT copy fails nothing further is touched (documented limitation: the nested arrays cannot be recovered and leak) and the copy error is returned. Device affinity: the model must be freed with its allocating device current — a mismatch returns cudaErrorInvalidDevice and frees nothing. Cleanup continues past a failed cudaFree; the FIRST error is returned.
     *
     * @param d_robotModel is a pointer returned by init_robotModel[_checked] (or nullptr)
     * @param failed_op (optional) receives a static string naming the failed operation
     * @return cudaSuccess or the first error
     */
    template <typename T>
    __host__
    cudaError_t free_robotModel_checked(robotModel<T> *d_robotModel, const char **failed_op = nullptr) {
        if (d_robotModel == nullptr) { return cudaSuccess; }
        cudaPointerAttributes attr; int current_device = -1;
        cudaError_t e = GRID_CUDA_CALL(cudaPointerGetAttributes(&attr, d_robotModel));
        if (e != cudaSuccess) { return grid_fail(failed_op, "cudaPointerGetAttributes(d_robotModel)", e); }
        e = GRID_CUDA_CALL(cudaGetDevice(&current_device));
        if (e != cudaSuccess) { return grid_fail(failed_op, "cudaGetDevice", e); }
        if (attr.type != cudaMemoryTypeDevice || attr.device != current_device) { return grid_fail(failed_op, "device affinity (d_robotModel was allocated on another device)", cudaErrorInvalidDevice); }
        robotModel<T> h_robotModel = {};
        e = GRID_CUDA_CALL(cudaMemcpy(&h_robotModel, d_robotModel, sizeof(robotModel<T>), cudaMemcpyDeviceToHost));
        if (e != cudaSuccess) { return grid_fail(failed_op, "cudaMemcpy(robotModel D2H; nested arrays unrecoverable)", e); }
        const char *cleanup_op = nullptr;
        cudaError_t first = release_robotModel_members<T>(h_robotModel, &cleanup_op);
        if (first != cudaSuccess) { grid_fail(failed_op, cleanup_op, first); }
        grid_cleanup_free(d_robotModel, "cudaFree(d_robotModel)", &first, failed_op != nullptr && *failed_op == nullptr ? failed_op : nullptr);
        return first;
    }

    /**
     * Frees a robotModel allocated by init_robotModel (legacy policy: exit on failure, or sticky first error under GRID_GPUERRCHK_NO_EXIT; prefer free_robotModel_checked in library code). A bare cudaFree(d_robotModel) frees ONLY the struct and leaks the nested arrays; this recovers them by copying the struct back to host first.
     *
     * @param d_robotModel is a pointer returned by init_robotModel
     */
    template <typename T>
    __host__
    void free_robotModel(robotModel<T> *d_robotModel) {
        const char *op = nullptr;
        cudaError_t e = free_robotModel_checked<T>(d_robotModel, &op);  // sequenced BEFORE reading op
        grid_legacy_check(e, op, __FILE__, __LINE__);
    }

    // Owning handle for a robotModel<T>: noncopyable, movable; init() constructs via
    // init_robotModel_checked, free() destroys via free_robotModel_checked (reportable),
    // the destructor destroys best-effort, release() transfers ownership to the caller.
    template <typename T>
    struct robotModel_owner {
        robotModel<T> *model = nullptr;
        robotModel_owner() = default;
        robotModel_owner(const robotModel_owner&) = delete;
        robotModel_owner& operator=(const robotModel_owner&) = delete;
        robotModel_owner(robotModel_owner &&o) noexcept : model(o.model) { o.model = nullptr; }
        robotModel_owner& operator=(robotModel_owner &&o) noexcept { if (this != &o) { free(); model = o.model; o.model = nullptr; } return *this; }
        ~robotModel_owner() { free(); }
        __host__ cudaError_t init(const char **failed_op = nullptr) { free(); return init_robotModel_checked<T>(&model, failed_op); }
        __host__ cudaError_t free(const char **failed_op = nullptr) { robotModel<T> *m = model; model = nullptr; return free_robotModel_checked<T>(m, failed_op); }
        __host__ robotModel<T>* release() { robotModel<T> *m = model; model = nullptr; return m; }
        __host__ robotModel<T>* get() const { return model; }
    };
    
    template <typename T>
    __host__
    T *grid_host_alloc(size_t bytes) {
        void *p = nullptr;
        if (cudaMallocHost(&p, bytes) == cudaSuccess) { return (T *)p; }
        cudaGetLastError(); // consume the failed pinned alloc
        return (T *)malloc(bytes);
    }
    
    template <typename T>
    __host__
    void grid_host_free(T *p) {
        if (p == nullptr) { return; }
        cudaPointerAttributes _attr;
        if (cudaPointerGetAttributes(&_attr, p) == cudaSuccess && _attr.type == cudaMemoryTypeHost) {
            cudaFreeHost(p);
            return;
        }
        cudaGetLastError();
        free(p);
    }
    
    struct grid_device_pool_t { void *base; size_t bytes; size_t used; int ws_slots; };
    // ⚠hidden visibility is LOAD-BEARING: without it the dynamic linker
    // unifies this inline function's static (weak symbol) across every
    // dlopened robot .so, so a second robot's init would carve from the
    // FIRST robot's (already exhausted) slab and 'OOM' on an empty GPU
    // (observed 2026-09-09, jax-then-torch two-robot process).
    __host__ inline __attribute__((visibility("hidden"))) grid_device_pool_t &grid_device_pool() {
        static grid_device_pool_t p = {nullptr, 0, 0, 0};
        return p;
    }
    __host__ __device__ constexpr size_t grid_pool_align(size_t b) { return (b + 255) & ~(size_t)255; }
    // W04-B B1 (K1): the allocator takes its pool EXPLICITLY so several arenas (runtime
    // contexts) on one .so never share a cursor; the pool-less overloads below keep the
    // historical one-liners (HJCD/GATO consumers) on the default pool — same caller API.
    __host__ inline cudaError_t grid_device_alloc(grid_device_pool_t *pool, void **p, size_t bytes) {
        if (pool != nullptr && pool->base != nullptr) {
            const size_t need = grid_pool_align(bytes);
            if (pool->used + need > pool->bytes) { *p = nullptr; return cudaErrorMemoryAllocation; }
            *p = (void *)((char *)pool->base + pool->used);
            pool->used += need;
            return cudaSuccess;
        }
        return cudaMalloc(p, bytes);
    }
    __host__ inline cudaError_t grid_device_alloc(void **p, size_t bytes) { return grid_device_alloc(&grid_device_pool(), p, bytes); }
    template <typename T>
    __host__ inline cudaError_t grid_device_free(grid_device_pool_t *pool, T *p) {
        if (pool != nullptr && pool->base != nullptr && (void *)p >= pool->base && (char *)p < (char *)pool->base + pool->bytes) {
            return cudaSuccess;  // carved from the caller-owned slab: nothing to free
        }
        return cudaFree((void *)p);
    }
    template <typename T>
    __host__ inline cudaError_t grid_device_free(T *p) { return grid_device_free(&grid_device_pool(), p); }
    
    template <typename T>
    __host__ inline void grid_cleanup_device_free(grid_device_pool_t *pool, T *p, const char *op, cudaError_t *first_cleanup_code, const char **first_cleanup_op) {
        if (p == nullptr) { return; }
        cudaError_t e = GRID_CUDA_CALL(grid_device_free(pool, p));
        if (e != cudaSuccess && first_cleanup_code != nullptr && *first_cleanup_code == cudaSuccess) {
            *first_cleanup_code = e; if (first_cleanup_op != nullptr) { *first_cleanup_op = op; }
        }
    }
    template <typename T>
    __host__ inline void grid_cleanup_device_free(T *p, const char *op, cudaError_t *first_cleanup_code, const char **first_cleanup_op) {
        grid_cleanup_device_free(&grid_device_pool(), p, op, first_cleanup_code, first_cleanup_op);
    }
    
    /**
     * Releases every device/host buffer a gridData owns (best effort, null members skipped; pool-carved buffers are rewound by the caller); the struct itself is NOT freed
     *
     * @param hd_data allocated by init_gridData[_checked] (or nullptr)
     * @return the first cudaFree error (cudaSuccess if none), named in *cleanup_op
     */
    template <typename T, gridDataKind KIND = GRID_DATA_ALL>
    __host__
    cudaError_t release_gridData_members(gridData<T, KIND> *hd_data, const char **cleanup_op = nullptr) {
        cudaError_t first = cudaSuccess;
        if (hd_data == nullptr) { return first; }
        const bool needs_dynamics = KIND == GRID_DATA_ALL || KIND == GRID_DATA_DYNAMICS;
        const bool needs_kinematics = KIND == GRID_DATA_ALL || KIND == GRID_DATA_KINEMATICS;
        // input variables used by dynamics and/or kinematics
        if (needs_dynamics || needs_kinematics) {
            grid_cleanup_device_free(hd_data->pool, hd_data->d_q_qd_u, "grid_device_free(d_q_qd_u)", &first, cleanup_op); hd_data->d_q_qd_u = nullptr;
            grid_cleanup_device_free(hd_data->pool, hd_data->d_q, "grid_device_free(d_q)", &first, cleanup_op); hd_data->d_q = nullptr;
            grid_host_free(hd_data->h_q_qd_u); hd_data->h_q_qd_u = nullptr;
            grid_host_free(hd_data->h_q); hd_data->h_q = nullptr;
            // external forces (body-major 6*NUM_BODIES local-frame); zeroed so the
            // default (no-fext) path subtracts nothing. Users overwrite h_f_ext and
            // copy to d_f_ext to apply external forces.
            grid_cleanup_device_free(hd_data->pool, hd_data->d_f_ext, "grid_device_free(d_f_ext)", &first, cleanup_op); hd_data->d_f_ext = nullptr;
            grid_host_free(hd_data->h_f_ext); hd_data->h_f_ext = nullptr;
        }
        if (needs_dynamics) {
            grid_cleanup_device_free(hd_data->pool, hd_data->d_q_qd, "grid_device_free(d_q_qd)", &first, cleanup_op); hd_data->d_q_qd = nullptr;
            grid_host_free(hd_data->h_q_qd); hd_data->h_q_qd = nullptr;
        }
        // dynamics outputs and fallback workspace
        if (needs_dynamics) {
            grid_cleanup_device_free(hd_data->pool, hd_data->d_c, "grid_device_free(d_c)", &first, cleanup_op); hd_data->d_c = nullptr;
            grid_cleanup_device_free(hd_data->pool, hd_data->d_Minv, "grid_device_free(d_Minv)", &first, cleanup_op); hd_data->d_Minv = nullptr;
            grid_cleanup_device_free(hd_data->pool, hd_data->d_qdd, "grid_device_free(d_qdd)", &first, cleanup_op); hd_data->d_qdd = nullptr;
            grid_cleanup_device_free(hd_data->pool, hd_data->d_M, "grid_device_free(d_M)", &first, cleanup_op); hd_data->d_M = nullptr;
            #if GRID_HAS_INVERSE_DYNAMICS_GRADIENT
            grid_cleanup_device_free(hd_data->pool, hd_data->d_dc_du, "grid_device_free(d_dc_du)", &first, cleanup_op); hd_data->d_dc_du = nullptr;
            #endif
            #if GRID_HAS_FORWARD_DYNAMICS_GRADIENT
            grid_cleanup_device_free(hd_data->pool, hd_data->d_df_du, "grid_device_free(d_df_du)", &first, cleanup_op); hd_data->d_df_du = nullptr;
            #endif
            // f_ext gradient column (section A): dtau/dfext, dqdd/dfext are each nv x (6*NB)
            #if GRID_HAS_F_EXT_GRADIENT
            grid_cleanup_device_free(hd_data->pool, hd_data->d_dtau_dfext, "grid_device_free(d_dtau_dfext)", &first, cleanup_op); hd_data->d_dtau_dfext = nullptr;
            grid_cleanup_device_free(hd_data->pool, hd_data->d_dqdd_dfext, "grid_device_free(d_dqdd_dfext)", &first, cleanup_op); hd_data->d_dqdd_dfext = nullptr;
            #endif
            grid_host_free(hd_data->h_dtau_dfext); hd_data->h_dtau_dfext = nullptr;
            grid_host_free(hd_data->h_dqdd_dfext); hd_data->h_dqdd_dfext = nullptr;
            // f_ext A.3: -dJ^T/dq = d(inverse_dynamics_gradient)/dfext, nv*6NB*nv (fixed base only; the largest per-timestep buffer)
            // sizeof(T) leads so the byte count is size_t throughout: the element count
            // alone overflows int on big robots (h2_plus nv=81 @N=1024: 3.06e9 > INT_MAX)
            #if GRID_HAS_F_EXT_GRADIENT_DQ
            grid_cleanup_device_free(hd_data->pool, hd_data->d_f_ext_gradient_dq, "grid_device_free(d_f_ext_gradient_dq)", &first, cleanup_op); hd_data->d_f_ext_gradient_dq = nullptr;
            grid_host_free(hd_data->h_f_ext_gradient_dq); hd_data->h_f_ext_gradient_dq = nullptr;
            #endif
            // R2: regressor Y and FD param-gradient dqdd/dpi (each nv x 10*NUM_BODIES)
            #if GRID_HAS_INVERSE_DYNAMICS_REGRESSOR
            grid_cleanup_device_free(hd_data->pool, hd_data->d_Y, "grid_device_free(d_Y)", &first, cleanup_op); hd_data->d_Y = nullptr;
            grid_host_free(hd_data->h_Y); hd_data->h_Y = nullptr;
            #endif
            #if GRID_HAS_FORWARD_DYNAMICS_PARAMETER_GRADIENT
            grid_cleanup_device_free(hd_data->pool, hd_data->d_dqdd_dpi, "grid_device_free(d_dqdd_dpi)", &first, cleanup_op); hd_data->d_dqdd_dpi = nullptr;
            grid_host_free(hd_data->h_dqdd_dpi); hd_data->h_dqdd_dpi = nullptr;
            #endif
            // B.0: dY/dx (dq | dqd halves, each direction an nv x 10NB row-major block).
            // sizeof(T) leads: 2*nv*nv*10NB*NUM_TIMESTEPS alone overflows int on big robots.
            #if GRID_HAS_INVERSE_DYNAMICS_REGRESSOR_GRADIENT
            grid_cleanup_device_free(hd_data->pool, hd_data->d_dY_dx, "grid_device_free(d_dY_dx)", &first, cleanup_op); hd_data->d_dY_dx = nullptr;
            grid_host_free(hd_data->h_dY_dx); hd_data->h_dY_dx = nullptr;
            #endif
            #if GRID_HAS_IDSVA_SO
            grid_cleanup_device_free(hd_data->pool, hd_data->d_idsva_so, "grid_device_free(d_idsva_so)", &first, cleanup_op); hd_data->d_idsva_so = nullptr;
            #endif
            #if GRID_HAS_FDSVA_SO
            grid_cleanup_device_free(hd_data->pool, hd_data->d_df2, "grid_device_free(d_df2)", &first, cleanup_op); hd_data->d_df2 = nullptr;
            #endif
            grid_host_free(hd_data->h_c); hd_data->h_c = nullptr;
            grid_host_free(hd_data->h_Minv); hd_data->h_Minv = nullptr;
            grid_host_free(hd_data->h_M); hd_data->h_M = nullptr;
            grid_host_free(hd_data->h_qdd); hd_data->h_qdd = nullptr;
            #if GRID_HAS_INVERSE_DYNAMICS_GRADIENT
            grid_host_free(hd_data->h_dc_du); hd_data->h_dc_du = nullptr;
            #endif
            #if GRID_HAS_FORWARD_DYNAMICS_GRADIENT
            grid_host_free(hd_data->h_df_du); hd_data->h_df_du = nullptr;
            #endif
            #if GRID_HAS_IDSVA_SO
            grid_host_free(hd_data->h_idsva_so); hd_data->h_idsva_so = nullptr;
            #endif
            #if GRID_HAS_FDSVA_SO
            grid_host_free(hd_data->h_df2); hd_data->h_df2 = nullptr;
            #endif
            #if GRID_HAS_INTEGRATOR
            grid_cleanup_device_free(hd_data->pool, hd_data->d_x_kp1, "grid_device_free(d_x_kp1)", &first, cleanup_op); hd_data->d_x_kp1 = nullptr;
            grid_host_free(hd_data->h_x_kp1); hd_data->h_x_kp1 = nullptr;
            #endif
            #if GRID_HAS_INTEGRATOR_GRADIENT
            grid_cleanup_device_free(hd_data->pool, hd_data->d_dAB, "grid_device_free(d_dAB)", &first, cleanup_op); hd_data->d_dAB = nullptr;
            grid_host_free(hd_data->h_dAB); hd_data->h_dAB = nullptr;
            #endif
        }
        // kinematics outputs
        if (needs_kinematics) {
            grid_cleanup_device_free(hd_data->pool, hd_data->d_end_effector_pose, "grid_device_free(d_end_effector_pose)", &first, cleanup_op); hd_data->d_end_effector_pose = nullptr;
            grid_cleanup_device_free(hd_data->pool, hd_data->d_end_effector_pose_gradient, "grid_device_free(d_end_effector_pose_gradient)", &first, cleanup_op); hd_data->d_end_effector_pose_gradient = nullptr;
            grid_cleanup_device_free(hd_data->pool, hd_data->d_end_effector_pose_hessian, "grid_device_free(d_end_effector_pose_hessian)", &first, cleanup_op); hd_data->d_end_effector_pose_hessian = nullptr;
            grid_host_free(hd_data->h_end_effector_pose); hd_data->h_end_effector_pose = nullptr;
            grid_host_free(hd_data->h_end_effector_pose_gradient); hd_data->h_end_effector_pose_gradient = nullptr;
            grid_host_free(hd_data->h_end_effector_pose_hessian); hd_data->h_end_effector_pose_hessian = nullptr;
            grid_cleanup_device_free(hd_data->pool, hd_data->d_frame_jacobian, "grid_device_free(d_frame_jacobian)", &first, cleanup_op); hd_data->d_frame_jacobian = nullptr;
            grid_cleanup_device_free(hd_data->pool, hd_data->d_frame_jacobian_dot, "grid_device_free(d_frame_jacobian_dot)", &first, cleanup_op); hd_data->d_frame_jacobian_dot = nullptr;
            grid_cleanup_device_free(hd_data->pool, hd_data->d_osc_inertia, "grid_device_free(d_osc_inertia)", &first, cleanup_op); hd_data->d_osc_inertia = nullptr;
            grid_host_free(hd_data->h_frame_jacobian); hd_data->h_frame_jacobian = nullptr;
            grid_host_free(hd_data->h_frame_jacobian_dot); hd_data->h_frame_jacobian_dot = nullptr;
            grid_host_free(hd_data->h_osc_inertia); hd_data->h_osc_inertia = nullptr;
            grid_cleanup_device_free(hd_data->pool, hd_data->d_eePose, "grid_device_free(d_eePose)", &first, cleanup_op); hd_data->d_eePose = nullptr;
            grid_cleanup_device_free(hd_data->pool, hd_data->d_eePoseGrad, "grid_device_free(d_eePoseGrad)", &first, cleanup_op); hd_data->d_eePoseGrad = nullptr;
            grid_cleanup_device_free(hd_data->pool, hd_data->d_eepose_runtime_offset, "grid_device_free(d_eepose_runtime_offset)", &first, cleanup_op); hd_data->d_eepose_runtime_offset = nullptr;
            grid_host_free(hd_data->h_eePose); hd_data->h_eePose = nullptr;
            grid_host_free(hd_data->h_eePoseGrad); hd_data->h_eePoseGrad = nullptr;
        }
        // G2 centroidal quick-wins outputs (com: 3+3*NV ; ccrba: 6*NV+6 ; energy: 3)
        if (needs_dynamics || needs_kinematics) {
            grid_cleanup_device_free(hd_data->pool, hd_data->d_com, "grid_device_free(d_com)", &first, cleanup_op); hd_data->d_com = nullptr;
            grid_cleanup_device_free(hd_data->pool, hd_data->d_ccrba, "grid_device_free(d_ccrba)", &first, cleanup_op); hd_data->d_ccrba = nullptr;
            grid_cleanup_device_free(hd_data->pool, hd_data->d_energy, "grid_device_free(d_energy)", &first, cleanup_op); hd_data->d_energy = nullptr;
            grid_host_free(hd_data->h_com); hd_data->h_com = nullptr;
            grid_host_free(hd_data->h_ccrba); hd_data->h_ccrba = nullptr;
            grid_host_free(hd_data->h_energy); hd_data->h_energy = nullptr;
            // PS5 energy regressors (each 10*NUM_BODIES): KE (dynamics) + PE (kinematics)
            grid_cleanup_device_free(hd_data->pool, hd_data->d_ke_regressor, "grid_device_free(d_ke_regressor)", &first, cleanup_op); hd_data->d_ke_regressor = nullptr;
            grid_cleanup_device_free(hd_data->pool, hd_data->d_pe_regressor, "grid_device_free(d_pe_regressor)", &first, cleanup_op); hd_data->d_pe_regressor = nullptr;
            grid_host_free(hd_data->h_ke_regressor); hd_data->h_ke_regressor = nullptr;
            grid_host_free(hd_data->h_pe_regressor); hd_data->h_pe_regressor = nullptr;
            // PS5 Coriolis matrix C(q,qd) (nv x nv)
            grid_cleanup_device_free(hd_data->pool, hd_data->d_coriolis, "grid_device_free(d_coriolis)", &first, cleanup_op); hd_data->d_coriolis = nullptr;
            grid_host_free(hd_data->h_coriolis); hd_data->h_coriolis = nullptr;
            // PS5 dCCRBA: dccrba tensor (6*nv*nv) + cmm_time_variation Adot (6*nv)
            grid_cleanup_device_free(hd_data->pool, hd_data->d_dccrba, "grid_device_free(d_dccrba)", &first, cleanup_op); hd_data->d_dccrba = nullptr;
            grid_cleanup_device_free(hd_data->pool, hd_data->d_cmm_time_variation, "grid_device_free(d_cmm_time_variation)", &first, cleanup_op); hd_data->d_cmm_time_variation = nullptr;
            grid_host_free(hd_data->h_dccrba); hd_data->h_dccrba = nullptr;
            grid_host_free(hd_data->h_cmm_time_variation); hd_data->h_cmm_time_variation = nullptr;
        }
        // workspace arena LAST: auto-fit slots to remaining device memory (see struct field).
            if (needs_dynamics || (needs_kinematics && (GRID_END_EFFECTOR_POSE_HESSIAN_USES_WORKSPACE_TEMP || GRID_END_EFFECTOR_POSE_GRADIENT_USES_WORKSPACE_TEMP || GRID_DCCRBA_USES_WORKSPACE_TEMP || GRID_OSC_INERTIA_USES_WORKSPACE))) {
                grid_cleanup_device_free(hd_data->pool, hd_data->d_workspace, "grid_device_free(d_workspace)", &first, cleanup_op); hd_data->d_workspace = nullptr;
                // Phase 3a/b/c/e: L2-pin d_workspace for its lifetime. Spilled buffers
                // (Minv-F, FD's Minv-F, ABA's inner scratch, FDSVA_SO's df_du/Minv) are
                // recursion-hot — L2 pinning narrows the smem→HBM gap to smem→L2.
            }
        return first;
    }

    /**
     * Library-safe allocation of the device and host memory for all computations: stops at the first failed allocation/copy, releases everything this attempt acquired, names the failed operation and publishes *out on complete success only (never exit/abort/cudaDeviceReset)
     *
     * @param out receives the gridData pointer (nullptr on failure)
     * @param failed_op (optional) receives a static string naming the failed operation
     * @return cudaSuccess or the first error
     */
    template <typename T, int NUM_TIMESTEPS, gridDataKind KIND = GRID_DATA_ALL>
    __host__
    cudaError_t init_gridData_checked(gridData<T, KIND> **out, const char **failed_op = nullptr, grid_device_pool_t *pool = nullptr) {
        grid_device_pool_t *_pool = (pool != nullptr) ? pool : &grid_device_pool();
        *out = nullptr;
        gridData<T, KIND> *hd_data = (gridData<T, KIND> *)GRID_HOST_ALLOC(calloc(1, sizeof(gridData<T, KIND>)));
        if (hd_data == nullptr) { return grid_fail(failed_op, "calloc(gridData)", cudaErrorMemoryAllocation); }
        hd_data->pool = _pool;
        const bool needs_dynamics = KIND == GRID_DATA_ALL || KIND == GRID_DATA_DYNAMICS;
        const bool needs_kinematics = KIND == GRID_DATA_ALL || KIND == GRID_DATA_KINEMATICS;
        // input variables used by dynamics and/or kinematics
        if (needs_dynamics || needs_kinematics) {
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_q_qd_u, 3*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_q_qd_u)", _e); } }
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_q, NUM_JOINTS*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_q)", _e); } }
            hd_data->h_q_qd_u = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(3*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_q_qd_u == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_q_qd_u)", cudaErrorMemoryAllocation); }
            hd_data->h_q = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(NUM_JOINTS*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_q == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_q)", cudaErrorMemoryAllocation); }
            // external forces (body-major 6*NUM_BODIES local-frame); zeroed so the
            // default (no-fext) path subtracts nothing. Users overwrite h_f_ext and
            // copy to d_f_ext to apply external forces.
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_f_ext, 6*NUM_BODIES*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_f_ext)", _e); } }
            { cudaError_t _e = GRID_CUDA_CALL(cudaMemset(hd_data->d_f_ext, 0, 6*NUM_BODIES*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "cudaMemset(d_f_ext)", _e); } }
            hd_data->h_f_ext = (T *)GRID_HOST_ALLOC((T *)calloc(6*NUM_BODIES*NUM_TIMESTEPS, sizeof(T))); if (hd_data->h_f_ext == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_f_ext)", cudaErrorMemoryAllocation); }
        }
        if (needs_dynamics) {
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_q_qd, 2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_q_qd)", _e); } }
            hd_data->h_q_qd = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_q_qd == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_q_qd)", cudaErrorMemoryAllocation); }
        }
        // dynamics outputs and fallback workspace
        if (needs_dynamics) {
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_c, NUM_JOINTS*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_c)", _e); } }
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_Minv, NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_Minv)", _e); } }
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_qdd, NUM_JOINTS*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_qdd)", _e); } }
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_M, NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_M)", _e); } }
            #if GRID_HAS_INVERSE_DYNAMICS_GRADIENT
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_dc_du, 2*NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_dc_du)", _e); } }
            #endif
            #if GRID_HAS_FORWARD_DYNAMICS_GRADIENT
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_df_du, 2*NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_df_du)", _e); } }
            #endif
            // f_ext gradient column (section A): dtau/dfext, dqdd/dfext are each nv x (6*NB)
            #if GRID_HAS_F_EXT_GRADIENT
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_dtau_dfext, NUM_VEL*6*NUM_BODIES*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_dtau_dfext)", _e); } }
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_dqdd_dfext, NUM_VEL*6*NUM_BODIES*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_dqdd_dfext)", _e); } }
            #endif
            hd_data->h_dtau_dfext = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(NUM_VEL*6*NUM_BODIES*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_dtau_dfext == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_dtau_dfext)", cudaErrorMemoryAllocation); }
            hd_data->h_dqdd_dfext = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(NUM_VEL*6*NUM_BODIES*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_dqdd_dfext == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_dqdd_dfext)", cudaErrorMemoryAllocation); }
            // f_ext A.3: -dJ^T/dq = d(inverse_dynamics_gradient)/dfext, nv*6NB*nv (fixed base only; the largest per-timestep buffer)
            // sizeof(T) leads so the byte count is size_t throughout: the element count
            // alone overflows int on big robots (h2_plus nv=81 @N=1024: 3.06e9 > INT_MAX)
            #if GRID_HAS_F_EXT_GRADIENT_DQ
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_f_ext_gradient_dq, sizeof(T)*NUM_VEL*6*NUM_BODIES*NUM_VEL*NUM_TIMESTEPS)); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_f_ext_gradient_dq)", _e); } }
            hd_data->h_f_ext_gradient_dq = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(sizeof(T)*NUM_VEL*6*NUM_BODIES*NUM_VEL*NUM_TIMESTEPS)); if (hd_data->h_f_ext_gradient_dq == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_f_ext_gradient_dq)", cudaErrorMemoryAllocation); }
            #endif
            // R2: regressor Y and FD param-gradient dqdd/dpi (each nv x 10*NUM_BODIES)
            #if GRID_HAS_INVERSE_DYNAMICS_REGRESSOR
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_Y, NUM_VEL*10*NUM_BODIES*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_Y)", _e); } }
            hd_data->h_Y = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(NUM_VEL*10*NUM_BODIES*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_Y == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_Y)", cudaErrorMemoryAllocation); }
            #endif
            #if GRID_HAS_FORWARD_DYNAMICS_PARAMETER_GRADIENT
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_dqdd_dpi, NUM_VEL*10*NUM_BODIES*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_dqdd_dpi)", _e); } }
            hd_data->h_dqdd_dpi = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(NUM_VEL*10*NUM_BODIES*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_dqdd_dpi == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_dqdd_dpi)", cudaErrorMemoryAllocation); }
            #endif
            // B.0: dY/dx (dq | dqd halves, each direction an nv x 10NB row-major block).
            // sizeof(T) leads: 2*nv*nv*10NB*NUM_TIMESTEPS alone overflows int on big robots.
            #if GRID_HAS_INVERSE_DYNAMICS_REGRESSOR_GRADIENT
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_dY_dx, sizeof(T)*2*NUM_VEL*NUM_VEL*10*NUM_BODIES*NUM_TIMESTEPS)); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_dY_dx)", _e); } }
            hd_data->h_dY_dx = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(sizeof(T)*2*NUM_VEL*NUM_VEL*10*NUM_BODIES*NUM_TIMESTEPS)); if (hd_data->h_dY_dx == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_dY_dx)", cudaErrorMemoryAllocation); }
            #endif
            #if GRID_HAS_IDSVA_SO
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_idsva_so, sizeof(T)*SECOND_ORDER_TENSOR_SIZE*NUM_TIMESTEPS)); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_idsva_so)", _e); } }
            #endif
            #if GRID_HAS_FDSVA_SO
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_df2, sizeof(T)*SECOND_ORDER_TENSOR_SIZE*NUM_TIMESTEPS)); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_df2)", _e); } }
            #endif
            hd_data->h_c = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(NUM_JOINTS*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_c == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_c)", cudaErrorMemoryAllocation); }
            hd_data->h_Minv = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_Minv == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_Minv)", cudaErrorMemoryAllocation); }
            hd_data->h_M = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_M == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_M)", cudaErrorMemoryAllocation); }
            hd_data->h_qdd = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(NUM_JOINTS*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_qdd == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_qdd)", cudaErrorMemoryAllocation); }
            #if GRID_HAS_INVERSE_DYNAMICS_GRADIENT
            hd_data->h_dc_du = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(2*NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_dc_du == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_dc_du)", cudaErrorMemoryAllocation); }
            #endif
            #if GRID_HAS_FORWARD_DYNAMICS_GRADIENT
            hd_data->h_df_du = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(2*NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_df_du == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_df_du)", cudaErrorMemoryAllocation); }
            #endif
            #if GRID_HAS_IDSVA_SO
            hd_data->h_idsva_so = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(sizeof(T)*SECOND_ORDER_TENSOR_SIZE*NUM_TIMESTEPS)); if (hd_data->h_idsva_so == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_idsva_so)", cudaErrorMemoryAllocation); }
            #endif
            #if GRID_HAS_FDSVA_SO
            hd_data->h_df2 = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(sizeof(T)*SECOND_ORDER_TENSOR_SIZE*NUM_TIMESTEPS)); if (hd_data->h_df2 == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_df2)", cudaErrorMemoryAllocation); }
            #endif
            #if GRID_HAS_INTEGRATOR
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_x_kp1, 2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_x_kp1)", _e); } }
            hd_data->h_x_kp1 = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_x_kp1 == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_x_kp1)", cudaErrorMemoryAllocation); }
            #endif
            #if GRID_HAS_INTEGRATOR_GRADIENT
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_dAB, 2*NUM_JOINTS*3*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_dAB)", _e); } }
            hd_data->h_dAB = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(2*NUM_JOINTS*3*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_dAB == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_dAB)", cudaErrorMemoryAllocation); }
            #endif
        }
        // kinematics outputs
        if (needs_kinematics) {
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_end_effector_pose, 6*NUM_EES*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_end_effector_pose)", _e); } }
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_end_effector_pose_gradient, 6*NUM_EES*NUM_VEL*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_end_effector_pose_gradient)", _e); } }
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_end_effector_pose_hessian, 6*NUM_EES*NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_end_effector_pose_hessian)", _e); } }
            hd_data->h_end_effector_pose = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(6*NUM_EES*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_end_effector_pose == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_end_effector_pose)", cudaErrorMemoryAllocation); }
            hd_data->h_end_effector_pose_gradient = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(6*NUM_EES*NUM_VEL*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_end_effector_pose_gradient == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_end_effector_pose_gradient)", cudaErrorMemoryAllocation); }
            hd_data->h_end_effector_pose_hessian = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(6*NUM_EES*NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_end_effector_pose_hessian == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_end_effector_pose_hessian)", cudaErrorMemoryAllocation); }
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_frame_jacobian, 6*NUM_VEL*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_frame_jacobian)", _e); } }
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_frame_jacobian_dot, 6*NUM_VEL*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_frame_jacobian_dot)", _e); } }
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_osc_inertia, 36*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_osc_inertia)", _e); } }
            hd_data->h_frame_jacobian = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(6*NUM_VEL*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_frame_jacobian == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_frame_jacobian)", cudaErrorMemoryAllocation); }
            hd_data->h_frame_jacobian_dot = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(6*NUM_VEL*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_frame_jacobian_dot == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_frame_jacobian_dot)", cudaErrorMemoryAllocation); }
            hd_data->h_osc_inertia = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(36*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_osc_inertia == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_osc_inertia)", cudaErrorMemoryAllocation); }
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_eePose, 6*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_eePose)", _e); } }
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_eePoseGrad, 6*NUM_VEL*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_eePoseGrad)", _e); } }
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_eepose_runtime_offset, 16*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_eepose_runtime_offset)", _e); } }
            { T h_Xtool_identity[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
              { cudaError_t _e = GRID_CUDA_CALL(cudaMemcpy(hd_data->d_eepose_runtime_offset, h_Xtool_identity, 16*sizeof(T), cudaMemcpyHostToDevice)); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "cudaMemcpy(d_eepose_runtime_offset)", _e); } } }
            hd_data->h_eePose = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(6*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_eePose == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_eePose)", cudaErrorMemoryAllocation); }
            hd_data->h_eePoseGrad = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(6*NUM_VEL*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_eePoseGrad == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_eePoseGrad)", cudaErrorMemoryAllocation); }
        }
        // G2 centroidal quick-wins outputs (com: 3+3*NV ; ccrba: 6*NV+6 ; energy: 3)
        if (needs_dynamics || needs_kinematics) {
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_com, (3+3*NUM_VEL)*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_com)", _e); } }
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_ccrba, (6*NUM_VEL+6)*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_ccrba)", _e); } }
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_energy, 3*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_energy)", _e); } }
            hd_data->h_com = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>((3+3*NUM_VEL)*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_com == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_com)", cudaErrorMemoryAllocation); }
            hd_data->h_ccrba = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>((6*NUM_VEL+6)*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_ccrba == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_ccrba)", cudaErrorMemoryAllocation); }
            hd_data->h_energy = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(3*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_energy == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_energy)", cudaErrorMemoryAllocation); }
            // PS5 energy regressors (each 10*NUM_BODIES): KE (dynamics) + PE (kinematics)
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_ke_regressor, 10*NUM_BODIES*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_ke_regressor)", _e); } }
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_pe_regressor, 10*NUM_BODIES*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_pe_regressor)", _e); } }
            hd_data->h_ke_regressor = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(10*NUM_BODIES*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_ke_regressor == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_ke_regressor)", cudaErrorMemoryAllocation); }
            hd_data->h_pe_regressor = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(10*NUM_BODIES*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_pe_regressor == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_pe_regressor)", cudaErrorMemoryAllocation); }
            // PS5 Coriolis matrix C(q,qd) (nv x nv)
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_coriolis, NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_coriolis)", _e); } }
            hd_data->h_coriolis = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_coriolis == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_coriolis)", cudaErrorMemoryAllocation); }
            // PS5 dCCRBA: dccrba tensor (6*nv*nv) + cmm_time_variation Adot (6*nv)
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_dccrba, 6*NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_dccrba)", _e); } }
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_cmm_time_variation, 6*NUM_VEL*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_cmm_time_variation)", _e); } }
            hd_data->h_dccrba = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(6*NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_dccrba == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_dccrba)", cudaErrorMemoryAllocation); }
            hd_data->h_cmm_time_variation = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(6*NUM_VEL*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_cmm_time_variation == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_cmm_time_variation)", cudaErrorMemoryAllocation); }
        }
        // workspace arena LAST: auto-fit slots to remaining device memory (see struct field).
            if (needs_dynamics || (needs_kinematics && (GRID_END_EFFECTOR_POSE_HESSIAN_USES_WORKSPACE_TEMP || GRID_END_EFFECTOR_POSE_GRADIENT_USES_WORKSPACE_TEMP || GRID_DCCRBA_USES_WORKSPACE_TEMP || GRID_OSC_INERTIA_USES_WORKSPACE))) {
                const size_t _ws_per_ts = GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()*GRID_WORKSPACE_SLOTS;
                int _ws_slots = NUM_TIMESTEPS;
                const char *_ws_env = getenv("GRID_WORKSPACE_TIMESTEP_SLOTS");
                if (_pool->ws_slots > 0) { _ws_slots = _pool->ws_slots < NUM_TIMESTEPS ? _pool->ws_slots : NUM_TIMESTEPS; }
                else if (_ws_env != nullptr && atoi(_ws_env) > 0) { _ws_slots = atoi(_ws_env) < NUM_TIMESTEPS ? atoi(_ws_env) : NUM_TIMESTEPS; }
                else if (_ws_per_ts > 0) {
                    size_t _ws_free = 0, _ws_total = 0;
                    { cudaError_t _e = GRID_CUDA_CALL(cudaMemGetInfo(&_ws_free, &_ws_total)); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "cudaMemGetInfo()", _e); } }
                    const size_t _ws_budget = _ws_free - _ws_free/10;  // 10% headroom
                    if (_ws_per_ts*(size_t)NUM_TIMESTEPS > _ws_budget) {
                        _ws_slots = (int)(_ws_budget/_ws_per_ts);
                        if (_ws_slots < 1) { _ws_slots = 1; }  // one slot must fit; else the malloc below fails loudly
                    }
                }
                hd_data->workspace_timestep_slots = _ws_slots;
                { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_workspace, _ws_per_ts*(size_t)_ws_slots)); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_workspace)", _e); } }
                // Phase 3a/b/c/e: L2-pin d_workspace for its lifetime. Spilled buffers
                // (Minv-F, FD's Minv-F, ABA's inner scratch, FDSVA_SO's df_du/Minv) are
                // recursion-hot — L2 pinning narrows the smem→HBM gap to smem→L2.
                { cudaError_t _e = GRID_CUDA_CALL(grid_begin_l2_persisting(0, hd_data->d_workspace, _ws_per_ts*(size_t)_ws_slots)); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_begin_l2_persisting(d_workspace)", _e); } }
            }
        *out = hd_data;
        return cudaSuccess;
    }

    template <typename T, gridDataKind KIND = GRID_DATA_ALL>
    __host__
    cudaError_t init_gridData_checked(int NUM_TIMESTEPS, gridData<T, KIND> **out, const char **failed_op = nullptr, grid_device_pool_t *pool = nullptr) {
        grid_device_pool_t *_pool = (pool != nullptr) ? pool : &grid_device_pool();
        *out = nullptr;
        gridData<T, KIND> *hd_data = (gridData<T, KIND> *)GRID_HOST_ALLOC(calloc(1, sizeof(gridData<T, KIND>)));
        if (hd_data == nullptr) { return grid_fail(failed_op, "calloc(gridData)", cudaErrorMemoryAllocation); }
        hd_data->pool = _pool;
        const bool needs_dynamics = KIND == GRID_DATA_ALL || KIND == GRID_DATA_DYNAMICS;
        const bool needs_kinematics = KIND == GRID_DATA_ALL || KIND == GRID_DATA_KINEMATICS;
        // input variables used by dynamics and/or kinematics
        if (needs_dynamics || needs_kinematics) {
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_q_qd_u, 3*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_q_qd_u)", _e); } }
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_q, NUM_JOINTS*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_q)", _e); } }
            hd_data->h_q_qd_u = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(3*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_q_qd_u == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_q_qd_u)", cudaErrorMemoryAllocation); }
            hd_data->h_q = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(NUM_JOINTS*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_q == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_q)", cudaErrorMemoryAllocation); }
            // external forces (body-major 6*NUM_BODIES local-frame); zeroed so the
            // default (no-fext) path subtracts nothing. Users overwrite h_f_ext and
            // copy to d_f_ext to apply external forces.
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_f_ext, 6*NUM_BODIES*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_f_ext)", _e); } }
            { cudaError_t _e = GRID_CUDA_CALL(cudaMemset(hd_data->d_f_ext, 0, 6*NUM_BODIES*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "cudaMemset(d_f_ext)", _e); } }
            hd_data->h_f_ext = (T *)GRID_HOST_ALLOC((T *)calloc(6*NUM_BODIES*NUM_TIMESTEPS, sizeof(T))); if (hd_data->h_f_ext == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_f_ext)", cudaErrorMemoryAllocation); }
        }
        if (needs_dynamics) {
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_q_qd, 2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_q_qd)", _e); } }
            hd_data->h_q_qd = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_q_qd == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_q_qd)", cudaErrorMemoryAllocation); }
        }
        // dynamics outputs and fallback workspace
        if (needs_dynamics) {
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_c, NUM_JOINTS*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_c)", _e); } }
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_Minv, NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_Minv)", _e); } }
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_qdd, NUM_JOINTS*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_qdd)", _e); } }
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_M, NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_M)", _e); } }
            #if GRID_HAS_INVERSE_DYNAMICS_GRADIENT
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_dc_du, 2*NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_dc_du)", _e); } }
            #endif
            #if GRID_HAS_FORWARD_DYNAMICS_GRADIENT
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_df_du, 2*NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_df_du)", _e); } }
            #endif
            // f_ext gradient column (section A): dtau/dfext, dqdd/dfext are each nv x (6*NB)
            #if GRID_HAS_F_EXT_GRADIENT
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_dtau_dfext, NUM_VEL*6*NUM_BODIES*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_dtau_dfext)", _e); } }
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_dqdd_dfext, NUM_VEL*6*NUM_BODIES*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_dqdd_dfext)", _e); } }
            #endif
            hd_data->h_dtau_dfext = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(NUM_VEL*6*NUM_BODIES*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_dtau_dfext == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_dtau_dfext)", cudaErrorMemoryAllocation); }
            hd_data->h_dqdd_dfext = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(NUM_VEL*6*NUM_BODIES*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_dqdd_dfext == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_dqdd_dfext)", cudaErrorMemoryAllocation); }
            // f_ext A.3: -dJ^T/dq = d(inverse_dynamics_gradient)/dfext, nv*6NB*nv (fixed base only; the largest per-timestep buffer)
            // sizeof(T) leads so the byte count is size_t throughout: the element count
            // alone overflows int on big robots (h2_plus nv=81 @N=1024: 3.06e9 > INT_MAX)
            #if GRID_HAS_F_EXT_GRADIENT_DQ
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_f_ext_gradient_dq, sizeof(T)*NUM_VEL*6*NUM_BODIES*NUM_VEL*NUM_TIMESTEPS)); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_f_ext_gradient_dq)", _e); } }
            hd_data->h_f_ext_gradient_dq = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(sizeof(T)*NUM_VEL*6*NUM_BODIES*NUM_VEL*NUM_TIMESTEPS)); if (hd_data->h_f_ext_gradient_dq == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_f_ext_gradient_dq)", cudaErrorMemoryAllocation); }
            #endif
            // R2: regressor Y and FD param-gradient dqdd/dpi (each nv x 10*NUM_BODIES)
            #if GRID_HAS_INVERSE_DYNAMICS_REGRESSOR
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_Y, NUM_VEL*10*NUM_BODIES*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_Y)", _e); } }
            hd_data->h_Y = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(NUM_VEL*10*NUM_BODIES*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_Y == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_Y)", cudaErrorMemoryAllocation); }
            #endif
            #if GRID_HAS_FORWARD_DYNAMICS_PARAMETER_GRADIENT
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_dqdd_dpi, NUM_VEL*10*NUM_BODIES*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_dqdd_dpi)", _e); } }
            hd_data->h_dqdd_dpi = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(NUM_VEL*10*NUM_BODIES*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_dqdd_dpi == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_dqdd_dpi)", cudaErrorMemoryAllocation); }
            #endif
            // B.0: dY/dx (dq | dqd halves, each direction an nv x 10NB row-major block).
            // sizeof(T) leads: 2*nv*nv*10NB*NUM_TIMESTEPS alone overflows int on big robots.
            #if GRID_HAS_INVERSE_DYNAMICS_REGRESSOR_GRADIENT
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_dY_dx, sizeof(T)*2*NUM_VEL*NUM_VEL*10*NUM_BODIES*NUM_TIMESTEPS)); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_dY_dx)", _e); } }
            hd_data->h_dY_dx = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(sizeof(T)*2*NUM_VEL*NUM_VEL*10*NUM_BODIES*NUM_TIMESTEPS)); if (hd_data->h_dY_dx == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_dY_dx)", cudaErrorMemoryAllocation); }
            #endif
            #if GRID_HAS_IDSVA_SO
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_idsva_so, sizeof(T)*SECOND_ORDER_TENSOR_SIZE*NUM_TIMESTEPS)); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_idsva_so)", _e); } }
            #endif
            #if GRID_HAS_FDSVA_SO
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_df2, sizeof(T)*SECOND_ORDER_TENSOR_SIZE*NUM_TIMESTEPS)); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_df2)", _e); } }
            #endif
            hd_data->h_c = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(NUM_JOINTS*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_c == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_c)", cudaErrorMemoryAllocation); }
            hd_data->h_Minv = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_Minv == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_Minv)", cudaErrorMemoryAllocation); }
            hd_data->h_M = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_M == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_M)", cudaErrorMemoryAllocation); }
            hd_data->h_qdd = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(NUM_JOINTS*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_qdd == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_qdd)", cudaErrorMemoryAllocation); }
            #if GRID_HAS_INVERSE_DYNAMICS_GRADIENT
            hd_data->h_dc_du = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(2*NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_dc_du == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_dc_du)", cudaErrorMemoryAllocation); }
            #endif
            #if GRID_HAS_FORWARD_DYNAMICS_GRADIENT
            hd_data->h_df_du = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(2*NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_df_du == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_df_du)", cudaErrorMemoryAllocation); }
            #endif
            #if GRID_HAS_IDSVA_SO
            hd_data->h_idsva_so = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(sizeof(T)*SECOND_ORDER_TENSOR_SIZE*NUM_TIMESTEPS)); if (hd_data->h_idsva_so == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_idsva_so)", cudaErrorMemoryAllocation); }
            #endif
            #if GRID_HAS_FDSVA_SO
            hd_data->h_df2 = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(sizeof(T)*SECOND_ORDER_TENSOR_SIZE*NUM_TIMESTEPS)); if (hd_data->h_df2 == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_df2)", cudaErrorMemoryAllocation); }
            #endif
            #if GRID_HAS_INTEGRATOR
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_x_kp1, 2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_x_kp1)", _e); } }
            hd_data->h_x_kp1 = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_x_kp1 == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_x_kp1)", cudaErrorMemoryAllocation); }
            #endif
            #if GRID_HAS_INTEGRATOR_GRADIENT
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_dAB, 2*NUM_JOINTS*3*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_dAB)", _e); } }
            hd_data->h_dAB = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(2*NUM_JOINTS*3*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_dAB == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_dAB)", cudaErrorMemoryAllocation); }
            #endif
        }
        // kinematics outputs
        if (needs_kinematics) {
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_end_effector_pose, 6*NUM_EES*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_end_effector_pose)", _e); } }
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_end_effector_pose_gradient, 6*NUM_EES*NUM_VEL*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_end_effector_pose_gradient)", _e); } }
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_end_effector_pose_hessian, 6*NUM_EES*NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_end_effector_pose_hessian)", _e); } }
            hd_data->h_end_effector_pose = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(6*NUM_EES*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_end_effector_pose == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_end_effector_pose)", cudaErrorMemoryAllocation); }
            hd_data->h_end_effector_pose_gradient = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(6*NUM_EES*NUM_VEL*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_end_effector_pose_gradient == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_end_effector_pose_gradient)", cudaErrorMemoryAllocation); }
            hd_data->h_end_effector_pose_hessian = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(6*NUM_EES*NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_end_effector_pose_hessian == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_end_effector_pose_hessian)", cudaErrorMemoryAllocation); }
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_frame_jacobian, 6*NUM_VEL*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_frame_jacobian)", _e); } }
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_frame_jacobian_dot, 6*NUM_VEL*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_frame_jacobian_dot)", _e); } }
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_osc_inertia, 36*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_osc_inertia)", _e); } }
            hd_data->h_frame_jacobian = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(6*NUM_VEL*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_frame_jacobian == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_frame_jacobian)", cudaErrorMemoryAllocation); }
            hd_data->h_frame_jacobian_dot = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(6*NUM_VEL*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_frame_jacobian_dot == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_frame_jacobian_dot)", cudaErrorMemoryAllocation); }
            hd_data->h_osc_inertia = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(36*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_osc_inertia == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_osc_inertia)", cudaErrorMemoryAllocation); }
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_eePose, 6*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_eePose)", _e); } }
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_eePoseGrad, 6*NUM_VEL*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_eePoseGrad)", _e); } }
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_eepose_runtime_offset, 16*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_eepose_runtime_offset)", _e); } }
            { T h_Xtool_identity[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
              { cudaError_t _e = GRID_CUDA_CALL(cudaMemcpy(hd_data->d_eepose_runtime_offset, h_Xtool_identity, 16*sizeof(T), cudaMemcpyHostToDevice)); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "cudaMemcpy(d_eepose_runtime_offset)", _e); } } }
            hd_data->h_eePose = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(6*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_eePose == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_eePose)", cudaErrorMemoryAllocation); }
            hd_data->h_eePoseGrad = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(6*NUM_VEL*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_eePoseGrad == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_eePoseGrad)", cudaErrorMemoryAllocation); }
        }
        // G2 centroidal quick-wins outputs (com: 3+3*NV ; ccrba: 6*NV+6 ; energy: 3)
        if (needs_dynamics || needs_kinematics) {
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_com, (3+3*NUM_VEL)*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_com)", _e); } }
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_ccrba, (6*NUM_VEL+6)*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_ccrba)", _e); } }
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_energy, 3*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_energy)", _e); } }
            hd_data->h_com = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>((3+3*NUM_VEL)*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_com == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_com)", cudaErrorMemoryAllocation); }
            hd_data->h_ccrba = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>((6*NUM_VEL+6)*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_ccrba == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_ccrba)", cudaErrorMemoryAllocation); }
            hd_data->h_energy = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(3*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_energy == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_energy)", cudaErrorMemoryAllocation); }
            // PS5 energy regressors (each 10*NUM_BODIES): KE (dynamics) + PE (kinematics)
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_ke_regressor, 10*NUM_BODIES*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_ke_regressor)", _e); } }
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_pe_regressor, 10*NUM_BODIES*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_pe_regressor)", _e); } }
            hd_data->h_ke_regressor = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(10*NUM_BODIES*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_ke_regressor == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_ke_regressor)", cudaErrorMemoryAllocation); }
            hd_data->h_pe_regressor = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(10*NUM_BODIES*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_pe_regressor == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_pe_regressor)", cudaErrorMemoryAllocation); }
            // PS5 Coriolis matrix C(q,qd) (nv x nv)
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_coriolis, NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_coriolis)", _e); } }
            hd_data->h_coriolis = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_coriolis == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_coriolis)", cudaErrorMemoryAllocation); }
            // PS5 dCCRBA: dccrba tensor (6*nv*nv) + cmm_time_variation Adot (6*nv)
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_dccrba, 6*NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_dccrba)", _e); } }
            { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_cmm_time_variation, 6*NUM_VEL*NUM_TIMESTEPS*sizeof(T))); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_cmm_time_variation)", _e); } }
            hd_data->h_dccrba = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(6*NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_dccrba == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_dccrba)", cudaErrorMemoryAllocation); }
            hd_data->h_cmm_time_variation = (T *)GRID_HOST_ALLOC(grid_host_alloc<T>(6*NUM_VEL*NUM_TIMESTEPS*sizeof(T))); if (hd_data->h_cmm_time_variation == nullptr) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "host_alloc(h_cmm_time_variation)", cudaErrorMemoryAllocation); }
        }
        // workspace arena LAST: auto-fit slots to remaining device memory (see struct field).
            if (needs_dynamics || (needs_kinematics && (GRID_END_EFFECTOR_POSE_HESSIAN_USES_WORKSPACE_TEMP || GRID_END_EFFECTOR_POSE_GRADIENT_USES_WORKSPACE_TEMP || GRID_DCCRBA_USES_WORKSPACE_TEMP || GRID_OSC_INERTIA_USES_WORKSPACE))) {
                const size_t _ws_per_ts = GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()*GRID_WORKSPACE_SLOTS;
                int _ws_slots = NUM_TIMESTEPS;
                const char *_ws_env = getenv("GRID_WORKSPACE_TIMESTEP_SLOTS");
                if (_pool->ws_slots > 0) { _ws_slots = _pool->ws_slots < NUM_TIMESTEPS ? _pool->ws_slots : NUM_TIMESTEPS; }
                else if (_ws_env != nullptr && atoi(_ws_env) > 0) { _ws_slots = atoi(_ws_env) < NUM_TIMESTEPS ? atoi(_ws_env) : NUM_TIMESTEPS; }
                else if (_ws_per_ts > 0) {
                    size_t _ws_free = 0, _ws_total = 0;
                    { cudaError_t _e = GRID_CUDA_CALL(cudaMemGetInfo(&_ws_free, &_ws_total)); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "cudaMemGetInfo()", _e); } }
                    const size_t _ws_budget = _ws_free - _ws_free/10;  // 10% headroom
                    if (_ws_per_ts*(size_t)NUM_TIMESTEPS > _ws_budget) {
                        _ws_slots = (int)(_ws_budget/_ws_per_ts);
                        if (_ws_slots < 1) { _ws_slots = 1; }  // one slot must fit; else the malloc below fails loudly
                    }
                }
                hd_data->workspace_timestep_slots = _ws_slots;
                { cudaError_t _e = GRID_CUDA_CALL(grid_device_alloc(_pool, (void**)&hd_data->d_workspace, _ws_per_ts*(size_t)_ws_slots)); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_device_alloc(d_workspace)", _e); } }
                // Phase 3a/b/c/e: L2-pin d_workspace for its lifetime. Spilled buffers
                // (Minv-F, FD's Minv-F, ABA's inner scratch, FDSVA_SO's df_du/Minv) are
                // recursion-hot — L2 pinning narrows the smem→HBM gap to smem→L2.
                { cudaError_t _e = GRID_CUDA_CALL(grid_begin_l2_persisting(0, hd_data->d_workspace, _ws_per_ts*(size_t)_ws_slots)); if (_e != cudaSuccess) { release_gridData_members<T, KIND>(hd_data); free(hd_data); return grid_fail(failed_op, "grid_begin_l2_persisting(d_workspace)", _e); } }
            }
        *out = hd_data;
        return cudaSuccess;
    }

    /**
     * Allocated device and host memory for all computations (legacy policy: exit on failure, or sticky first error + nullptr under GRID_GPUERRCHK_NO_EXIT; prefer init_gridData_checked in library code)
     *
     * @return A pointer to the gridData struct of pointers
     */
    template <typename T, int NUM_TIMESTEPS, gridDataKind KIND = GRID_DATA_ALL>
    __host__
    gridData<T, KIND> *init_gridData(){
        gridData<T, KIND> *hd_data = nullptr; const char *op = nullptr;
        cudaError_t e = init_gridData_checked<T, NUM_TIMESTEPS, KIND>(&hd_data, &op);  // sequenced BEFORE reading op
        grid_legacy_check(e, op, __FILE__, __LINE__);
        return hd_data;
    }

    /**
     * Allocated device and host memory for all computations (legacy policy; prefer init_gridData_checked in library code)
     *
     * @param Max number of timesteps in the trajectory
     * @return A pointer to the gridData struct of pointers
     */
    template <typename T, gridDataKind KIND = GRID_DATA_ALL>
    __host__
    gridData<T, KIND> *init_gridData(int NUM_TIMESTEPS){
        gridData<T, KIND> *hd_data = nullptr; const char *op = nullptr;
        cudaError_t e = init_gridData_checked<T, KIND>(NUM_TIMESTEPS, &hd_data, &op);  // sequenced BEFORE reading op
        grid_legacy_check(e, op, __FILE__, __LINE__);
        return hd_data;
    }

    /**
     * Device bytes a pool-mode init_gridData will carve for this KIND at the given workspace slot count — size the slab handed to grid_device_pool() with this (derived from the SAME allocation list as init_gridData).
     *
     * @param workspace timestep slots (clamped to [1, NUM_TIMESTEPS])
     * @return total device bytes (256-aligned per allocation)
     */
    template <typename T, int NUM_TIMESTEPS, gridDataKind KIND = GRID_DATA_ALL>
    __host__
    size_t gridData_device_bytes(int ws_slots = NUM_TIMESTEPS){
        size_t _total = 0;
        const bool needs_dynamics = KIND == GRID_DATA_ALL || KIND == GRID_DATA_DYNAMICS;
        const bool needs_kinematics = KIND == GRID_DATA_ALL || KIND == GRID_DATA_KINEMATICS;
        if (needs_dynamics || needs_kinematics) {
            _total += grid_pool_align(3*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
            _total += grid_pool_align(NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
            _total += grid_pool_align(6*NUM_BODIES*NUM_TIMESTEPS*sizeof(T));
        }
        if (needs_dynamics) {
            _total += grid_pool_align(2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        }
        if (needs_dynamics) {
            _total += grid_pool_align(NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
            _total += grid_pool_align(NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T));
            _total += grid_pool_align(NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
            _total += grid_pool_align(NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T));
            #if GRID_HAS_INVERSE_DYNAMICS_GRADIENT
            _total += grid_pool_align(2*NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T));
            #endif
            #if GRID_HAS_FORWARD_DYNAMICS_GRADIENT
            _total += grid_pool_align(2*NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T));
            #endif
            #if GRID_HAS_F_EXT_GRADIENT
            _total += grid_pool_align(NUM_VEL*6*NUM_BODIES*NUM_TIMESTEPS*sizeof(T));
            _total += grid_pool_align(NUM_VEL*6*NUM_BODIES*NUM_TIMESTEPS*sizeof(T));
            #endif
            #if GRID_HAS_F_EXT_GRADIENT_DQ
            _total += grid_pool_align(sizeof(T)*NUM_VEL*6*NUM_BODIES*NUM_VEL*NUM_TIMESTEPS);
            #endif
            #if GRID_HAS_INVERSE_DYNAMICS_REGRESSOR
            _total += grid_pool_align(NUM_VEL*10*NUM_BODIES*NUM_TIMESTEPS*sizeof(T));
            #endif
            #if GRID_HAS_FORWARD_DYNAMICS_PARAMETER_GRADIENT
            _total += grid_pool_align(NUM_VEL*10*NUM_BODIES*NUM_TIMESTEPS*sizeof(T));
            #endif
            #if GRID_HAS_INVERSE_DYNAMICS_REGRESSOR_GRADIENT
            _total += grid_pool_align(sizeof(T)*2*NUM_VEL*NUM_VEL*10*NUM_BODIES*NUM_TIMESTEPS);
            #endif
            #if GRID_HAS_IDSVA_SO
            _total += grid_pool_align(sizeof(T)*SECOND_ORDER_TENSOR_SIZE*NUM_TIMESTEPS);
            #endif
            #if GRID_HAS_FDSVA_SO
            _total += grid_pool_align(sizeof(T)*SECOND_ORDER_TENSOR_SIZE*NUM_TIMESTEPS);
            #endif
            #if GRID_HAS_INVERSE_DYNAMICS_GRADIENT
            #endif
            #if GRID_HAS_FORWARD_DYNAMICS_GRADIENT
            #endif
            #if GRID_HAS_IDSVA_SO
            #endif
            #if GRID_HAS_FDSVA_SO
            #endif
            #if GRID_HAS_INTEGRATOR
            _total += grid_pool_align(2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
            #endif
            #if GRID_HAS_INTEGRATOR_GRADIENT
            _total += grid_pool_align(2*NUM_JOINTS*3*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
            #endif
        }
        if (needs_kinematics) {
            _total += grid_pool_align(6*NUM_EES*NUM_TIMESTEPS*sizeof(T));
            _total += grid_pool_align(6*NUM_EES*NUM_VEL*NUM_TIMESTEPS*sizeof(T));
            _total += grid_pool_align(6*NUM_EES*NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T));
            _total += grid_pool_align(6*NUM_VEL*NUM_TIMESTEPS*sizeof(T));
            _total += grid_pool_align(6*NUM_VEL*NUM_TIMESTEPS*sizeof(T));
            _total += grid_pool_align(36*NUM_TIMESTEPS*sizeof(T));
            _total += grid_pool_align(6*NUM_TIMESTEPS*sizeof(T));
            _total += grid_pool_align(6*NUM_VEL*NUM_TIMESTEPS*sizeof(T));
            _total += grid_pool_align(16*sizeof(T));
        }
        if (needs_dynamics || needs_kinematics) {
            _total += grid_pool_align((3+3*NUM_VEL)*NUM_TIMESTEPS*sizeof(T));
            _total += grid_pool_align((6*NUM_VEL+6)*NUM_TIMESTEPS*sizeof(T));
            _total += grid_pool_align(3*NUM_TIMESTEPS*sizeof(T));
            _total += grid_pool_align(10*NUM_BODIES*NUM_TIMESTEPS*sizeof(T));
            _total += grid_pool_align(10*NUM_BODIES*NUM_TIMESTEPS*sizeof(T));
            _total += grid_pool_align(NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T));
            _total += grid_pool_align(6*NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T));
            _total += grid_pool_align(6*NUM_VEL*NUM_TIMESTEPS*sizeof(T));
        }
            if (needs_dynamics || (needs_kinematics && (GRID_END_EFFECTOR_POSE_HESSIAN_USES_WORKSPACE_TEMP || GRID_END_EFFECTOR_POSE_GRADIENT_USES_WORKSPACE_TEMP || GRID_DCCRBA_USES_WORKSPACE_TEMP || GRID_OSC_INERTIA_USES_WORKSPACE))) {
                const size_t _ws_per_ts = GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()*GRID_WORKSPACE_SLOTS;
                int _ws_slots = ws_slots < 1 ? 1 : (ws_slots < NUM_TIMESTEPS ? ws_slots : NUM_TIMESTEPS);
                _total += grid_pool_align(_ws_per_ts*(size_t)_ws_slots);
            }
        return _total;
    }

    /**
     * Initializes joint limits (lower/upper) in GPU memory
     *
     * Notes:
     *   Memory order is lower[0..n-1], upper[0..n-1]
     *
     * @return A device pointer to the joint limits array
     */
    template <typename T>
    __host__
    cudaError_t init_joint_limits_checked(T **out, const char **failed_op = nullptr) {
        *out = nullptr;
        T h_joint_limits[14];
        h_joint_limits[0] = static_cast<T>(-2.96706);
        h_joint_limits[7] = static_cast<T>(2.96706);
        h_joint_limits[1] = static_cast<T>(-2.0944);
        h_joint_limits[8] = static_cast<T>(2.0944);
        h_joint_limits[2] = static_cast<T>(-2.96706);
        h_joint_limits[9] = static_cast<T>(2.96706);
        h_joint_limits[3] = static_cast<T>(-2.0944);
        h_joint_limits[10] = static_cast<T>(2.0944);
        h_joint_limits[4] = static_cast<T>(-2.96706);
        h_joint_limits[11] = static_cast<T>(2.96706);
        h_joint_limits[5] = static_cast<T>(-2.0944);
        h_joint_limits[12] = static_cast<T>(2.0944);
        h_joint_limits[6] = static_cast<T>(-3.05433);
        h_joint_limits[13] = static_cast<T>(3.05433);
        T *d_joint_limits = nullptr;
        cudaError_t _e = GRID_CUDA_CALL(cudaMalloc((void**)&d_joint_limits,14*sizeof(T)));
        if (_e != cudaSuccess) { return grid_fail(failed_op, "cudaMalloc(d_joint_limits)", _e); }
        _e = GRID_CUDA_CALL(cudaMemcpy(d_joint_limits,h_joint_limits,14*sizeof(T),cudaMemcpyHostToDevice));
        
        if (_e != cudaSuccess) { grid_cleanup_free(d_joint_limits, "cudaFree(d_joint_limits)", nullptr, nullptr); return grid_fail(failed_op, "cudaMemcpy(d_joint_limits)", _e); }
        *out = d_joint_limits;
        return cudaSuccess;
    }

    template <typename T>
    __host__
    T* init_joint_limits() {
        T *d = nullptr; const char *op = nullptr;
        cudaError_t e = init_joint_limits_checked<T>(&d, &op);  // sequenced BEFORE reading op
        grid_legacy_check(e, op, __FILE__, __LINE__);
        return d;
    }

    /**
     * Updates the Xmats in (shared) GPU memory acording to the configuration
     *
     * @param s_XImats is the (shared) memory destination location for the XImats
     * @param s_q is the (shared) memory location of the current configuration
     * @param s_topology_helpers is the (shared) memory location for the topology_helpers (nullptr/unused for serial chains with identical Ss)
     * @param d_robotModel is the pointer to the initialized model specific helpers (XImats, mxfuncs, topology_helpers, etc.)
     * @param s_temp is temporary (shared) memory used to compute sin and cos if needed of size: 14
     */
    template <typename T, bool SKIP_FLOATING_BASE_X = false>
    __device__ __forceinline__
    void load_update_XImats_helpers(T *s_XImats, const T *s_q, int *s_topology_helpers, const robotModel<T> *d_robotModel, T *s_temp) {
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 504; ind += blockDim.x*blockDim.y){
            s_XImats[ind] = d_robotModel->d_XImats[ind];
        }
        for(int k = threadIdx.x + threadIdx.y*blockDim.x; k < 7; k += blockDim.x*blockDim.y){
            s_temp[k] = static_cast<T>(sin(s_q[k]));
            s_temp[k+7] = static_cast<T>(cos(s_q[k]));
        }
        __syncthreads();
        if(threadIdx.x == 0 && threadIdx.y == 0){
            // X[0]
            s_XImats[0] = static_cast<T>(1.0*s_temp[7]);
            s_XImats[1] = static_cast<T>(-1.0*s_temp[0]);
            s_XImats[3] = static_cast<T>(-0.1575*s_temp[0]);
            s_XImats[4] = static_cast<T>(-0.1575*s_temp[7]);
            s_XImats[6] = static_cast<T>(1.0*s_temp[0]);
            s_XImats[7] = static_cast<T>(1.0*s_temp[7]);
            s_XImats[9] = static_cast<T>(0.1575*s_temp[7]);
            s_XImats[10] = static_cast<T>(-0.1575*s_temp[0]);
            // X[1]
            s_XImats[36] = static_cast<T>(s_temp[1]);
            s_XImats[37] = static_cast<T>(s_temp[8]);
            s_XImats[42] = static_cast<T>(1.60982338570648e-15*s_temp[1]);
            s_XImats[43] = static_cast<T>(1.60982338570648e-15*s_temp[8]);
            s_XImats[45] = static_cast<T>(0.2025*s_temp[1]);
            s_XImats[46] = static_cast<T>(0.2025*s_temp[8]);
            s_XImats[48] = static_cast<T>(s_temp[8]);
            s_XImats[49] = static_cast<T>(-s_temp[1]);
            // X[2]
            s_XImats[78] = static_cast<T>(s_temp[9]);
            s_XImats[79] = static_cast<T>(-s_temp[2]);
            s_XImats[81] = static_cast<T>(-0.2045*s_temp[2]);
            s_XImats[82] = static_cast<T>(-0.2045*s_temp[9]);
            s_XImats[84] = static_cast<T>(s_temp[2]);
            s_XImats[85] = static_cast<T>(s_temp[9]);
            s_XImats[87] = static_cast<T>(0.2045*s_temp[9]);
            s_XImats[88] = static_cast<T>(-0.2045*s_temp[2]);
            // X[3]
            s_XImats[108] = static_cast<T>(s_temp[10]);
            s_XImats[109] = static_cast<T>(-s_temp[3]);
            s_XImats[117] = static_cast<T>(0.2155*s_temp[10]);
            s_XImats[118] = static_cast<T>(-0.2155*s_temp[3]);
            s_XImats[120] = static_cast<T>(s_temp[3]);
            s_XImats[121] = static_cast<T>(s_temp[10]);
            // X[4]
            s_XImats[144] = static_cast<T>(s_temp[11]);
            s_XImats[145] = static_cast<T>(-s_temp[4]);
            s_XImats[147] = static_cast<T>(-0.1845*s_temp[4]);
            s_XImats[148] = static_cast<T>(-0.1845*s_temp[11]);
            s_XImats[156] = static_cast<T>(-s_temp[4]);
            s_XImats[157] = static_cast<T>(-s_temp[11]);
            s_XImats[159] = static_cast<T>(-0.1845*s_temp[11]);
            s_XImats[160] = static_cast<T>(0.1845*s_temp[4]);
            // X[5]
            s_XImats[180] = static_cast<T>(s_temp[5]);
            s_XImats[181] = static_cast<T>(s_temp[12]);
            s_XImats[183] = static_cast<T>(-0.0607*s_temp[12]);
            s_XImats[184] = static_cast<T>(0.0607*s_temp[5]);
            s_XImats[186] = static_cast<T>(1.60982338570648e-15*s_temp[5]);
            s_XImats[187] = static_cast<T>(1.60982338570648e-15*s_temp[12]);
            s_XImats[189] = static_cast<T>(0.2155*s_temp[5]);
            s_XImats[190] = static_cast<T>(0.2155*s_temp[12]);
            s_XImats[192] = static_cast<T>(s_temp[12]);
            s_XImats[193] = static_cast<T>(-s_temp[5]);
            s_XImats[195] = static_cast<T>(0.0607*s_temp[5]);
            s_XImats[196] = static_cast<T>(0.0607*s_temp[12]);
            // X[6]
            s_XImats[219] = static_cast<T>(-0.0607*s_temp[13]);
            s_XImats[220] = static_cast<T>(0.0607*s_temp[6]);
            s_XImats[222] = static_cast<T>(s_temp[13]);
            s_XImats[223] = static_cast<T>(-s_temp[6]);
            s_XImats[225] = static_cast<T>(-0.081*s_temp[6]);
            s_XImats[226] = static_cast<T>(-0.081*s_temp[13]);
            s_XImats[228] = static_cast<T>(s_temp[6]);
            s_XImats[229] = static_cast<T>(s_temp[13]);
            s_XImats[231] = static_cast<T>(0.081*s_temp[13]);
            s_XImats[232] = static_cast<T>(-0.081*s_temp[6]);
        }
        __syncthreads();
        for(int kcr = threadIdx.x + threadIdx.y*blockDim.x; kcr < 63; kcr += blockDim.x*blockDim.y){
            int k = kcr / 9; int cr = kcr % 9; int c = cr / 3; int r = cr % 3;
            int srcInd = k*36 + c*6 + r; int dstInd = srcInd + 21; // 3 more rows and cols
            s_XImats[dstInd] = s_XImats[srcInd];
        }
        __syncthreads();
    }

    /**
     * Cooperatively fill s_topology_helpers from the model (no-op when TOPOLOGY_HELPERS_COUNT == 0); for external callers of the *_inner functions
     *
     * @param s_topology_helpers is the (shared) int buffer to fill (size TOPOLOGY_HELPERS_COUNT; nullptr/unused when 0)
     * @param d_robotModel holds d_topology_helpers
     */
    template <typename T>
    __device__ __forceinline__
    void load_topology_helpers(int *s_topology_helpers, const robotModel<T> *d_robotModel) {
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 0; ind += blockDim.x*blockDim.y){
            s_topology_helpers[ind] = d_robotModel->d_topology_helpers[ind];
        }
        __syncthreads();
    }

    /**
     * Updates the (d)XmatsHom in (shared) GPU memory acording to the configuration
     *
     * @param s_XmatsHom is the (shared) memory destination location for the XmatsHom
     * @param s_q is the (shared) memory location of the current configuration
     * @param s_topology_helpers is the (shared) memory location for the topology_helpers (nullptr/unused for serial chains with identical Ss)
     * @param d_robotModel is the pointer to the initialized model specific helpers (XImats, mxfuncs, topology_helpers, etc.)
     * @param s_temp is temporary (shared) memory used to compute sin and cos if needed of size: 14
     */
    template <typename T>
    __device__ __forceinline__
    void load_update_XmatsHom_helpers(T *s_XmatsHom, int *s_topology_helpers, const T *s_q, const robotModel<T> *d_robotModel, T *s_temp) {
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 144; ind += blockDim.x*blockDim.y){
            s_XmatsHom[ind] = d_robotModel->d_XImats[ind+504];
        }
        for(int k = threadIdx.x + threadIdx.y*blockDim.x; k < 7; k += blockDim.x*blockDim.y){
            s_temp[k] = static_cast<T>(sin(s_q[k]));
            s_temp[k+7] = static_cast<T>(cos(s_q[k]));
        }
        __syncthreads();
        if(threadIdx.x == 0 && threadIdx.y == 0){
            // X_hom[0]
            s_XmatsHom[0] = static_cast<T>(s_temp[7]);
            s_XmatsHom[1] = static_cast<T>(s_temp[0]);
            s_XmatsHom[4] = static_cast<T>(-s_temp[0]);
            s_XmatsHom[5] = static_cast<T>(s_temp[7]);
            // X_hom[1]
            s_XmatsHom[16] = static_cast<T>(s_temp[1]);
            s_XmatsHom[17] = static_cast<T>(1.6098233857064764e-15*s_temp[1]);
            s_XmatsHom[18] = static_cast<T>(s_temp[8]);
            s_XmatsHom[20] = static_cast<T>(s_temp[8]);
            s_XmatsHom[21] = static_cast<T>(1.6098233857064764e-15*s_temp[8]);
            s_XmatsHom[22] = static_cast<T>(-s_temp[1]);
            // X_hom[2]
            s_XmatsHom[33] = static_cast<T>(s_temp[9]);
            s_XmatsHom[34] = static_cast<T>(s_temp[2]);
            s_XmatsHom[37] = static_cast<T>(-s_temp[2]);
            s_XmatsHom[38] = static_cast<T>(s_temp[9]);
            // X_hom[3]
            s_XmatsHom[48] = static_cast<T>(s_temp[10]);
            s_XmatsHom[50] = static_cast<T>(s_temp[3]);
            s_XmatsHom[52] = static_cast<T>(-s_temp[3]);
            s_XmatsHom[54] = static_cast<T>(s_temp[10]);
            // X_hom[4]
            s_XmatsHom[64] = static_cast<T>(s_temp[11]);
            s_XmatsHom[66] = static_cast<T>(-s_temp[4]);
            s_XmatsHom[68] = static_cast<T>(-s_temp[4]);
            s_XmatsHom[70] = static_cast<T>(-s_temp[11]);
            // X_hom[5]
            s_XmatsHom[80] = static_cast<T>(s_temp[5]);
            s_XmatsHom[81] = static_cast<T>(1.6098233857064764e-15*s_temp[5]);
            s_XmatsHom[82] = static_cast<T>(s_temp[12]);
            s_XmatsHom[84] = static_cast<T>(s_temp[12]);
            s_XmatsHom[85] = static_cast<T>(1.6098233857064764e-15*s_temp[12]);
            s_XmatsHom[86] = static_cast<T>(-s_temp[5]);
            // X_hom[6]
            s_XmatsHom[97] = static_cast<T>(s_temp[13]);
            s_XmatsHom[98] = static_cast<T>(s_temp[6]);
            s_XmatsHom[101] = static_cast<T>(-s_temp[6]);
            s_XmatsHom[102] = static_cast<T>(s_temp[13]);
        }
        __syncthreads();
    }

    /**
     * Updates the (d)XmatsHom in (shared) GPU memory acording to the configuration
     *
     * @param s_XmatsHom is the (shared) memory destination location for the XmatsHom
     * @param s_dXmatsHom is the (shared) memory destination location for the dXmatsHom
     * @param s_q is the (shared) memory location of the current configuration
     * @param s_topology_helpers is the (shared) memory location for the topology_helpers (nullptr/unused for serial chains with identical Ss)
     * @param d_robotModel is the pointer to the initialized model specific helpers (XImats, mxfuncs, topology_helpers, etc.)
     * @param s_temp is temporary (shared) memory used to compute sin and cos if needed of size: 14
     */
    template <typename T>
    __device__ __forceinline__
    void load_update_XmatsHom_helpers(T *s_XmatsHom, T *s_dXmatsHom, int *s_topology_helpers, const T *s_q, const robotModel<T> *d_robotModel, T *s_temp) {
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 144; ind += blockDim.x*blockDim.y){
            s_XmatsHom[ind] = d_robotModel->d_XImats[ind+504];
        }
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 112; ind += blockDim.x*blockDim.y){
            s_dXmatsHom[ind] = d_robotModel->d_XImats[ind+648];
        }
        for(int k = threadIdx.x + threadIdx.y*blockDim.x; k < 7; k += blockDim.x*blockDim.y){
            s_temp[k] = static_cast<T>(sin(s_q[k]));
            s_temp[k+7] = static_cast<T>(cos(s_q[k]));
        }
        __syncthreads();
        if(threadIdx.x == 0 && threadIdx.y == 0){
            // X_hom[0]
            s_XmatsHom[0] = static_cast<T>(s_temp[7]);
            s_XmatsHom[1] = static_cast<T>(s_temp[0]);
            s_XmatsHom[4] = static_cast<T>(-s_temp[0]);
            s_XmatsHom[5] = static_cast<T>(s_temp[7]);
            // X_hom[1]
            s_XmatsHom[16] = static_cast<T>(s_temp[1]);
            s_XmatsHom[17] = static_cast<T>(1.6098233857064764e-15*s_temp[1]);
            s_XmatsHom[18] = static_cast<T>(s_temp[8]);
            s_XmatsHom[20] = static_cast<T>(s_temp[8]);
            s_XmatsHom[21] = static_cast<T>(1.6098233857064764e-15*s_temp[8]);
            s_XmatsHom[22] = static_cast<T>(-s_temp[1]);
            // X_hom[2]
            s_XmatsHom[33] = static_cast<T>(s_temp[9]);
            s_XmatsHom[34] = static_cast<T>(s_temp[2]);
            s_XmatsHom[37] = static_cast<T>(-s_temp[2]);
            s_XmatsHom[38] = static_cast<T>(s_temp[9]);
            // X_hom[3]
            s_XmatsHom[48] = static_cast<T>(s_temp[10]);
            s_XmatsHom[50] = static_cast<T>(s_temp[3]);
            s_XmatsHom[52] = static_cast<T>(-s_temp[3]);
            s_XmatsHom[54] = static_cast<T>(s_temp[10]);
            // X_hom[4]
            s_XmatsHom[64] = static_cast<T>(s_temp[11]);
            s_XmatsHom[66] = static_cast<T>(-s_temp[4]);
            s_XmatsHom[68] = static_cast<T>(-s_temp[4]);
            s_XmatsHom[70] = static_cast<T>(-s_temp[11]);
            // X_hom[5]
            s_XmatsHom[80] = static_cast<T>(s_temp[5]);
            s_XmatsHom[81] = static_cast<T>(1.6098233857064764e-15*s_temp[5]);
            s_XmatsHom[82] = static_cast<T>(s_temp[12]);
            s_XmatsHom[84] = static_cast<T>(s_temp[12]);
            s_XmatsHom[85] = static_cast<T>(1.6098233857064764e-15*s_temp[12]);
            s_XmatsHom[86] = static_cast<T>(-s_temp[5]);
            // X_hom[6]
            s_XmatsHom[97] = static_cast<T>(s_temp[13]);
            s_XmatsHom[98] = static_cast<T>(s_temp[6]);
            s_XmatsHom[101] = static_cast<T>(-s_temp[6]);
            s_XmatsHom[102] = static_cast<T>(s_temp[13]);
        }
        __syncthreads();
        if(threadIdx.x == 0 && threadIdx.y == 0){
            // dX_hom[0]
            s_dXmatsHom[0] = static_cast<T>(-s_temp[0]);
            s_dXmatsHom[1] = static_cast<T>(s_temp[7]);
            s_dXmatsHom[4] = static_cast<T>(-s_temp[7]);
            s_dXmatsHom[5] = static_cast<T>(-s_temp[0]);
            // dX_hom[1]
            s_dXmatsHom[16] = static_cast<T>(s_temp[8]);
            s_dXmatsHom[17] = static_cast<T>(1.6098233857064764e-15*s_temp[8]);
            s_dXmatsHom[18] = static_cast<T>(-s_temp[1]);
            s_dXmatsHom[20] = static_cast<T>(-s_temp[1]);
            s_dXmatsHom[21] = static_cast<T>(-1.6098233857064764e-15*s_temp[1]);
            s_dXmatsHom[22] = static_cast<T>(-s_temp[8]);
            // dX_hom[2]
            s_dXmatsHom[33] = static_cast<T>(-s_temp[2]);
            s_dXmatsHom[34] = static_cast<T>(s_temp[9]);
            s_dXmatsHom[37] = static_cast<T>(-s_temp[9]);
            s_dXmatsHom[38] = static_cast<T>(-s_temp[2]);
            // dX_hom[3]
            s_dXmatsHom[48] = static_cast<T>(-s_temp[3]);
            s_dXmatsHom[50] = static_cast<T>(s_temp[10]);
            s_dXmatsHom[52] = static_cast<T>(-s_temp[10]);
            s_dXmatsHom[54] = static_cast<T>(-s_temp[3]);
            // dX_hom[4]
            s_dXmatsHom[64] = static_cast<T>(-s_temp[4]);
            s_dXmatsHom[66] = static_cast<T>(-s_temp[11]);
            s_dXmatsHom[68] = static_cast<T>(-s_temp[11]);
            s_dXmatsHom[70] = static_cast<T>(s_temp[4]);
            // dX_hom[5]
            s_dXmatsHom[80] = static_cast<T>(s_temp[12]);
            s_dXmatsHom[81] = static_cast<T>(1.6098233857064764e-15*s_temp[12]);
            s_dXmatsHom[82] = static_cast<T>(-s_temp[5]);
            s_dXmatsHom[84] = static_cast<T>(-s_temp[5]);
            s_dXmatsHom[85] = static_cast<T>(-1.6098233857064764e-15*s_temp[5]);
            s_dXmatsHom[86] = static_cast<T>(-s_temp[12]);
            // dX_hom[6]
            s_dXmatsHom[97] = static_cast<T>(-s_temp[6]);
            s_dXmatsHom[98] = static_cast<T>(s_temp[13]);
            s_dXmatsHom[101] = static_cast<T>(-s_temp[13]);
            s_dXmatsHom[102] = static_cast<T>(-s_temp[6]);
        }
        __syncthreads();
    }

    /**
     * Computes the End Effector Position
     *
     * Notes:
     *   Assumes the Xhom matricies have already been updated for the given q
     *   Defaults to all leave nodes if fixed_target_name is not provided
     *
     * @param s_end_effector_pose is a pointer to shared memory of size 6*NUM_EE where NUM_EE = 1
     * @param s_q is the vector of joint positions
     * @param s_Xhom is the pointer to the homogenous transformation matricies 
     * @param s_temp is a pointer to helper shared memory of size 32
     * @param d_workspace is the global-memory chain workspace used in place of s_temp when !TEMP_IN_SMEM
     * @param s_topology_helpers is the (shared) memory location for the topology_helpers (nullptr/unused for serial chains with identical Ss)
     * @param s_linalg_smem is optional byte-addressed shared memory (reserved; unused by this inner)
     */
    template <typename T, bool TEMP_IN_SMEM = true>
    __device__
    void end_effector_pose_inner(T *s_end_effector_pose, const T *s_q, const T *s_Xhom, int *s_topology_helpers, T *s_temp, T *d_workspace, unsigned char *s_linalg_smem) {
        if constexpr (!TEMP_IN_SMEM) { s_temp = d_workspace; } else { (void)d_workspace; }
        //
        // For each branch in parallel chain up the transform
        // Keep chaining until reaching the root (starting from the leaves)
        //
        // Serial chain manipulator so optimize as parent is jid-1
        // First set to leaf (or fixed) transform
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 16; ind += blockDim.x*blockDim.y){
            s_temp[ind] = s_Xhom[16*6 + ind];
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 1/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 16; ind += blockDim.x*blockDim.y){
            int row = ind % 4; int col = ind / 4;
            s_temp[ind + 16] = dot_prod<T,4,4,1>(&s_Xhom[16*5 + row], &s_temp[0 + 4*col]);
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 2/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 16; ind += blockDim.x*blockDim.y){
            int row = ind % 4; int col = ind / 4;
            s_temp[ind + 0] = dot_prod<T,4,4,1>(&s_Xhom[16*4 + row], &s_temp[16 + 4*col]);
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 3/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 16; ind += blockDim.x*blockDim.y){
            int row = ind % 4; int col = ind / 4;
            s_temp[ind + 16] = dot_prod<T,4,4,1>(&s_Xhom[16*3 + row], &s_temp[0 + 4*col]);
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 4/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 16; ind += blockDim.x*blockDim.y){
            int row = ind % 4; int col = ind / 4;
            s_temp[ind + 0] = dot_prod<T,4,4,1>(&s_Xhom[16*2 + row], &s_temp[16 + 4*col]);
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 5/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 16; ind += blockDim.x*blockDim.y){
            int row = ind % 4; int col = ind / 4;
            s_temp[ind + 16] = dot_prod<T,4,4,1>(&s_Xhom[16*1 + row], &s_temp[0 + 4*col]);
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 6/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 16; ind += blockDim.x*blockDim.y){
            int row = ind % 4; int col = ind / 4;
            s_temp[ind + 0] = dot_prod<T,4,4,1>(&s_Xhom[16*0 + row], &s_temp[16 + 4*col]);
        }
        __syncthreads();
        //
        // Now extract the end_effector_pose from the transforms.
        // (This generic family evaluates the last MOVING joint; a terminal fixed
        // joint's <origin> is tracked by the named fixed-target family instead.)
        //
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 3; ind += blockDim.x*blockDim.y){
            // xyz is easy
            int xyzInd = ind % 3; int eeInd = ind / 3; T *s_Xmat_hom = &s_temp[0 + 16*eeInd];
            s_end_effector_pose[6*eeInd + xyzInd] = s_Xmat_hom[12 + xyzInd];
            // roll pitch yaw is a bit more difficult
            if(xyzInd > 0){continue;}
            s_end_effector_pose[6*eeInd + 3] = atan2(s_Xmat_hom[6],s_Xmat_hom[10]);
            s_end_effector_pose[6*eeInd + 4] = -atan2(s_Xmat_hom[2],sqrt(s_Xmat_hom[6]*s_Xmat_hom[6] + s_Xmat_hom[10]*s_Xmat_hom[10]));
            s_end_effector_pose[6*eeInd + 5] = atan2(s_Xmat_hom[1],s_Xmat_hom[0]);
        }
        __syncthreads();
    }

    /**
     * Computes the End Effector Position
     *
     * @param s_end_effector_pose is a pointer to shared memory of size 6*NUM_EE where NUM_EE = 1
     * @param s_q is the vector of joint positions
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     */
    template <typename T>
    __device__
    void end_effector_pose_device(T *s_end_effector_pose, const T *s_q, const robotModel<T> *d_robotModel) {
        // GRID shared arena layout
        //   T s_XmatsHom[144]
        //   T s_temp[32]
        //   bytes s_linalg_smem[GRID_EE_LINALG_SHARED_BYTES<T>()]
        extern __shared__ __align__(16) unsigned char s_arena[];
        size_t s_arena_offset = 0;
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_XmatsHom = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(144);
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_temp = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(32);
        int *s_topology_helpers = nullptr;
        unsigned char *s_linalg_smem = nullptr;
        if (static_cast<size_t>(GRID_EE_LINALG_SHARED_BYTES<T>()) > 0) {
            s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
            s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
            s_arena_offset += static_cast<size_t>(GRID_EE_LINALG_SHARED_BYTES<T>());
        }
        #ifdef GRID_CUDA_DEBUG_LAYOUT
        assert(s_arena_offset == grid_shared_arena_bytes<T>(176, 0, GRID_EE_LINALG_SHARED_BYTES<T>()));
        #endif
        (void)s_arena_offset;
        load_update_XmatsHom_helpers<T>(s_XmatsHom, s_topology_helpers, s_q, d_robotModel, s_temp);
        end_effector_pose_inner<T, true>(s_end_effector_pose, s_q, s_XmatsHom, s_topology_helpers, s_temp, nullptr, s_linalg_smem);
    }

    /**
     * Compute the End Effector Position
     *
     * @param d_end_effector_pose is the vector of end effector positions
     * @param d_q is the vector of joint positions
     * @param stride_q is the stide between each q
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER, bool MUJOCO_OUTPUT = false>
    __global__
    __launch_bounds__(tier_max_threads<RESOURCE_TIER>())
    void end_effector_pose_kernel_single_timing(T *d_end_effector_pose, const T *d_q, const int stride_q, const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS) {
        // GRID shared arena layout
        //   T s_q[7]
        //   T s_end_effector_pose[6]
        //   T s_XmatsHom[144]
        //   T s_temp[32]
        //   bytes s_linalg_smem[GRID_EE_LINALG_SHARED_BYTES<T>()]
        extern __shared__ __align__(16) unsigned char s_arena[];
        size_t s_arena_offset = 0;
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_q = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(7);
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_end_effector_pose = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(6);
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_XmatsHom = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(144);
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_temp = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(32);
        int *s_topology_helpers = nullptr;
        unsigned char *s_linalg_smem = nullptr;
        if (static_cast<size_t>(GRID_EE_LINALG_SHARED_BYTES<T>()) > 0) {
            s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
            s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
            s_arena_offset += static_cast<size_t>(GRID_EE_LINALG_SHARED_BYTES<T>());
        }
        #ifdef GRID_CUDA_DEBUG_LAYOUT
        assert(s_arena_offset == grid_shared_arena_bytes<T>(189, 0, GRID_EE_LINALG_SHARED_BYTES<T>()));
        #endif
        (void)s_arena_offset;
        // load to shared mem
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
            s_q[ind] = d_q[ind];
        }
        __syncthreads();
        // compute with NUM_TIMESTEPS as NUM_REPS for timing
        for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
            // anti-LICM: volatile reload of inputs each rep
            for(int _aopt_i = threadIdx.x + threadIdx.y*blockDim.x; _aopt_i < 7; _aopt_i += blockDim.x*blockDim.y){
                reinterpret_cast<volatile T *>(s_q)[_aopt_i] = reinterpret_cast<const volatile T *>(d_q)[_aopt_i];
            }
            __syncthreads();
            // anti-LICM (1/2): stomp one input slot with `rep`
            if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
                reinterpret_cast<volatile T *>(s_q)[rep % (7)] = static_cast<T>(rep);
            }
            // anti-LICM (2/2): feedback prev rep's d_end_effector_pose into s_q (true loop-carried dep)
            if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
                T _aopt_fb1 = reinterpret_cast<const volatile T *>(d_end_effector_pose)[(rep + 0x3FF) & 0x3FF];
                T _aopt_fb2 = reinterpret_cast<const volatile T *>(d_end_effector_pose)[(rep + 0x3FE) & 0x3FF];
                T _aopt_fb3 = reinterpret_cast<const volatile T *>(d_end_effector_pose)[(rep + 0x3FD) & 0x3FF];
                reinterpret_cast<volatile T *>(s_q)[(rep + 1) % (7)] += _aopt_fb1;
                reinterpret_cast<volatile T *>(s_q)[(rep + 2) % (7)] += _aopt_fb2;
                reinterpret_cast<volatile T *>(s_q)[(rep + 3) % (7)] += _aopt_fb3;
            }
            __syncthreads();
            load_update_XmatsHom_helpers<T>(s_XmatsHom, s_topology_helpers, s_q, d_robotModel, s_temp);
            end_effector_pose_inner<T, true>(s_end_effector_pose, s_q, s_XmatsHom, s_topology_helpers, s_temp, nullptr, s_linalg_smem);
            __syncthreads();
            if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) { reinterpret_cast<volatile T *>(d_end_effector_pose)[rep & 1023] = reinterpret_cast<const volatile T *>(s_end_effector_pose)[rep & 7]; }
        }
        // save down to global
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            d_end_effector_pose[ind] = s_end_effector_pose[ind];
        }
        __syncthreads();
    }

    /**
     * Compute the End Effector Position
     *
     * @param d_end_effector_pose is the vector of end effector positions
     * @param d_q is the vector of joint positions
     * @param stride_q is the stide between each q
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER, bool MUJOCO_OUTPUT = false>
    __global__
    __launch_bounds__(tier_max_threads<RESOURCE_TIER>())
    void end_effector_pose_kernel(T *d_end_effector_pose, const T *d_q, const int stride_q, const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS) {
        // GRID shared arena layout
        //   T s_q[7]
        //   T s_end_effector_pose[6]
        //   T s_XmatsHom[144]
        //   T s_temp[32]
        //   bytes s_linalg_smem[GRID_EE_LINALG_SHARED_BYTES<T>()]
        extern __shared__ __align__(16) unsigned char s_arena[];
        size_t s_arena_offset = 0;
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_q = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(7);
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_end_effector_pose = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(6);
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_XmatsHom = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(144);
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_temp = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(32);
        int *s_topology_helpers = nullptr;
        unsigned char *s_linalg_smem = nullptr;
        if (static_cast<size_t>(GRID_EE_LINALG_SHARED_BYTES<T>()) > 0) {
            s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
            s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
            s_arena_offset += static_cast<size_t>(GRID_EE_LINALG_SHARED_BYTES<T>());
        }
        #ifdef GRID_CUDA_DEBUG_LAYOUT
        assert(s_arena_offset == grid_shared_arena_bytes<T>(189, 0, GRID_EE_LINALG_SHARED_BYTES<T>()));
        #endif
        (void)s_arena_offset;
        for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
            // load to shared mem
            const T *d_q_k = &d_q[k*stride_q];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                s_q[ind] = d_q_k[ind];
            }
            __syncthreads();
            // compute
            load_update_XmatsHom_helpers<T>(s_XmatsHom, s_topology_helpers, s_q, d_robotModel, s_temp);
            end_effector_pose_inner<T, true>(s_end_effector_pose, s_q, s_XmatsHom, s_topology_helpers, s_temp, nullptr, s_linalg_smem);
            __syncthreads();
            // save down to global
            T *d_end_effector_pose_k = &d_end_effector_pose[k*6];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
                d_end_effector_pose_k[ind] = s_end_effector_pose[ind];
            }
            __syncthreads();
        }
    }

    /**
     * Compute the End Effector Pose
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_COMPRESSED_MEM = false, gridDataKind KIND = GRID_DATA_ALL, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>
    __host__
    void end_effector_pose(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps,
                                const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_KINEMATICS, "end_effector_pose requires all-data or kinematics gridData");
        // start code with memory transfer
        int stride_q;
        if (USE_COMPRESSED_MEM) {stride_q = NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q,hd_data->h_q,stride_q*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        else {stride_q = 3*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        gpuErrchkKernel();
        // then call the kernel
        gpuErrchk(grid_check_dynamic_shared_memory_bytes("end_effector_pose", END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_BYTES<T>()));
        dim3 _grid_thr_clamped_1 = grid_host_clamp_threads((const void*)&end_effector_pose_kernel<T, RESOURCE_TIER>, thread_dimms);
        if (USE_COMPRESSED_MEM) {end_effector_pose_kernel<T, RESOURCE_TIER><<<block_dimms,_grid_thr_clamped_1,END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_BYTES<T>()>>>(hd_data->d_end_effector_pose,hd_data->d_q,stride_q,d_robotModel,num_timesteps);}
        else                    {end_effector_pose_kernel<T, RESOURCE_TIER><<<block_dimms,_grid_thr_clamped_1,END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_BYTES<T>()>>>(hd_data->d_end_effector_pose,hd_data->d_q_qd_u,stride_q,d_robotModel,num_timesteps);}
        gpuErrchkKernel();
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_end_effector_pose,hd_data->d_end_effector_pose,6*NUM_EES*num_timesteps*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchkKernel();
    }

    /**
     * Compute the End Effector Pose
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_COMPRESSED_MEM = false, gridDataKind KIND = GRID_DATA_ALL, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>
    __host__
    void end_effector_pose_single_timing(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps,
                                              const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_KINEMATICS, "end_effector_pose requires all-data or kinematics gridData");
        // start code with memory transfer
        int stride_q;
        if (USE_COMPRESSED_MEM) {stride_q = NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q,hd_data->h_q,stride_q*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        else {stride_q = 3*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        gpuErrchkKernel();
        // then call the kernel
        gpuErrchk(grid_check_dynamic_shared_memory_bytes("end_effector_pose", END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_BYTES<T>()));
        struct timespec start, end; clock_gettime(CLOCK_MONOTONIC,&start);
        dim3 _grid_thr_clamped_2 = grid_host_clamp_threads((const void*)&end_effector_pose_kernel_single_timing<T, RESOURCE_TIER>, thread_dimms);
        if (USE_COMPRESSED_MEM) {end_effector_pose_kernel_single_timing<T, RESOURCE_TIER><<<block_dimms,_grid_thr_clamped_2,END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_BYTES<T>()>>>(hd_data->d_end_effector_pose,hd_data->d_q,stride_q,d_robotModel,num_timesteps);}
        else                    {end_effector_pose_kernel_single_timing<T, RESOURCE_TIER><<<block_dimms,_grid_thr_clamped_2,END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_BYTES<T>()>>>(hd_data->d_end_effector_pose,hd_data->d_q_qd_u,stride_q,d_robotModel,num_timesteps);}
        gpuErrchkKernel();
        clock_gettime(CLOCK_MONOTONIC,&end);
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_end_effector_pose,hd_data->d_end_effector_pose,6*NUM_EES*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchkKernel();
        printf("Single Call END_EFFECTOR_POSE %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(num_timesteps));
    }

    /**
     * Compute the End Effector Pose
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_COMPRESSED_MEM = false, gridDataKind KIND = GRID_DATA_ALL, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>
    __host__
    void end_effector_pose_compute_only(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps,
                                             const dim3 block_dimms, const dim3 thread_dimms) {
        static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_KINEMATICS, "end_effector_pose requires all-data or kinematics gridData");
        int stride_q = USE_COMPRESSED_MEM ? NUM_JOINTS: 3*NUM_JOINTS;
        // then call the kernel
        gpuErrchk(grid_check_dynamic_shared_memory_bytes("end_effector_pose", END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_BYTES<T>()));
        dim3 _grid_thr_clamped_3 = grid_host_clamp_threads((const void*)&end_effector_pose_kernel<T, RESOURCE_TIER>, thread_dimms);
        if (USE_COMPRESSED_MEM) {end_effector_pose_kernel<T, RESOURCE_TIER><<<block_dimms,_grid_thr_clamped_3,END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_BYTES<T>()>>>(hd_data->d_end_effector_pose,hd_data->d_q,stride_q,d_robotModel,num_timesteps);}
        else                    {end_effector_pose_kernel<T, RESOURCE_TIER><<<block_dimms,_grid_thr_clamped_3,END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_BYTES<T>()>>>(hd_data->d_end_effector_pose,hd_data->d_q_qd_u,stride_q,d_robotModel,num_timesteps);}
        gpuErrchkKernel();
    }

    /**
     * Computes the Gradient of the End Effector Pose with respect to generalized velocity (d/dv tangent, pinocchio convention)
     *
     * Notes:
     *   Assumes s_Xhom has been populated with the per-joint LOCAL transforms for the given q.
     *   Output d/dv (TANGENT) is 6 x nv per ee (was 6 x nq for d/dq) -- matches pinocchio.
     *
     * @param s_end_effector_pose_gradient is a pointer to shared memory of size 6*NUM_VEL*NUM_EE where NUM_VEL = 7 and NUM_EE = 1
     * @param s_q is the vector of joint positions (unused; kept for signature compatibility)
     * @param s_Xhom is the pointer to the LOCAL homogeneous transformation matrices (per-joint Xhom_local)
     * @param s_dXhom is the pointer to the LOCAL d-transforms (unused by the geometric-Jacobian path; kept for signature compatibility)
     * @param s_temp is a pointer to helper shared memory of size 190
     * @param d_workspace is the global-memory chain workspace used in place of s_temp when !TEMP_IN_SMEM
     * @param s_topology_helpers is the (shared) memory location for the topology_helpers (nullptr/unused for serial chains with identical Ss)
     * @param s_linalg_smem is optional byte-addressed shared memory (reserved; unused)
     */
    template <typename T, bool TEMP_IN_SMEM = true>
    __device__
    void end_effector_pose_gradient_inner(T *s_end_effector_pose_gradient, const T *s_q, const T *s_Xhom, const T *s_dXhom, int *s_topology_helpers, T *s_temp, T *d_workspace, unsigned char *s_linalg_smem) {
        if constexpr (!TEMP_IN_SMEM) { s_temp = d_workspace; } else { (void)d_workspace; }
        (void)s_q; (void)s_dXhom; (void)s_linalg_smem;
        // scratch layout: Xworld | Jv (3 x nv x ee) | Jw (3 x nv x ee) | E_sincos (4 x ee)
        T *s_Xworld = &s_temp[0];
        T *s_Jv     = &s_temp[144];
        T *s_Jw     = &s_temp[165];
        T *s_E_sc   = &s_temp[186];   // cy,sy,cp,sp per ee
        //
        // Step 1: build world transforms for every joint via BFS-level chain-up
        //
        // BFS level 0 -> joints [0]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 16; ind += blockDim.x*blockDim.y){
            int slot = ind / 16; int ele = ind % 16;
            int row = ele & 3; int col = ele >> 2;
            // branch to get pointer locations
            int jid; int par;
                 if (slot < 1){ jid = 0; par = -1; }
            if (par == -1) {
                s_Xworld[16*jid + ele] = s_Xhom[16*jid + ele];
            }
            else {
                s_Xworld[16*jid + ele] = dot_prod<T,4,4,1>(&s_Xworld[16*par + row], &s_Xhom[16*jid + 4*col]);
            }
        }
        __syncthreads();
        // BFS level 1 -> joints [1]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 16; ind += blockDim.x*blockDim.y){
            int slot = ind / 16; int ele = ind % 16;
            int row = ele & 3; int col = ele >> 2;
            // branch to get pointer locations
            int jid; int par;
                 if (slot < 1){ jid = 1; par = 0; }
            if (par == -1) {
                s_Xworld[16*jid + ele] = s_Xhom[16*jid + ele];
            }
            else {
                s_Xworld[16*jid + ele] = dot_prod<T,4,4,1>(&s_Xworld[16*par + row], &s_Xhom[16*jid + 4*col]);
            }
        }
        __syncthreads();
        // BFS level 2 -> joints [2]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 16; ind += blockDim.x*blockDim.y){
            int slot = ind / 16; int ele = ind % 16;
            int row = ele & 3; int col = ele >> 2;
            // branch to get pointer locations
            int jid; int par;
                 if (slot < 1){ jid = 2; par = 1; }
            if (par == -1) {
                s_Xworld[16*jid + ele] = s_Xhom[16*jid + ele];
            }
            else {
                s_Xworld[16*jid + ele] = dot_prod<T,4,4,1>(&s_Xworld[16*par + row], &s_Xhom[16*jid + 4*col]);
            }
        }
        __syncthreads();
        // BFS level 3 -> joints [3]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 16; ind += blockDim.x*blockDim.y){
            int slot = ind / 16; int ele = ind % 16;
            int row = ele & 3; int col = ele >> 2;
            // branch to get pointer locations
            int jid; int par;
                 if (slot < 1){ jid = 3; par = 2; }
            if (par == -1) {
                s_Xworld[16*jid + ele] = s_Xhom[16*jid + ele];
            }
            else {
                s_Xworld[16*jid + ele] = dot_prod<T,4,4,1>(&s_Xworld[16*par + row], &s_Xhom[16*jid + 4*col]);
            }
        }
        __syncthreads();
        // BFS level 4 -> joints [4]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 16; ind += blockDim.x*blockDim.y){
            int slot = ind / 16; int ele = ind % 16;
            int row = ele & 3; int col = ele >> 2;
            // branch to get pointer locations
            int jid; int par;
                 if (slot < 1){ jid = 4; par = 3; }
            if (par == -1) {
                s_Xworld[16*jid + ele] = s_Xhom[16*jid + ele];
            }
            else {
                s_Xworld[16*jid + ele] = dot_prod<T,4,4,1>(&s_Xworld[16*par + row], &s_Xhom[16*jid + 4*col]);
            }
        }
        __syncthreads();
        // BFS level 5 -> joints [5]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 16; ind += blockDim.x*blockDim.y){
            int slot = ind / 16; int ele = ind % 16;
            int row = ele & 3; int col = ele >> 2;
            // branch to get pointer locations
            int jid; int par;
                 if (slot < 1){ jid = 5; par = 4; }
            if (par == -1) {
                s_Xworld[16*jid + ele] = s_Xhom[16*jid + ele];
            }
            else {
                s_Xworld[16*jid + ele] = dot_prod<T,4,4,1>(&s_Xworld[16*par + row], &s_Xhom[16*jid + 4*col]);
            }
        }
        __syncthreads();
        // BFS level 6 -> joints [6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 16; ind += blockDim.x*blockDim.y){
            int slot = ind / 16; int ele = ind % 16;
            int row = ele & 3; int col = ele >> 2;
            // branch to get pointer locations
            int jid; int par;
                 if (slot < 1){ jid = 6; par = 5; }
            if (par == -1) {
                s_Xworld[16*jid + ele] = s_Xhom[16*jid + ele];
            }
            else {
                s_Xworld[16*jid + ele] = dot_prod<T,4,4,1>(&s_Xworld[16*par + row], &s_Xhom[16*jid + 4*col]);
            }
        }
        __syncthreads();
        //
        // Step 2: zero the J_v and J_w scratch (out-of-chain columns stay zero)
        //
        glass::set_const<T, 42>(static_cast<T>(0), s_Jv);
        //
        // Step 3: per-chain-joint columns of J_v, J_w (one block-parallel work-item per (ee, S-column))
        //
        static const int eeg_job_j[] = { 0, 1, 2, 3, 4, 5, 6 };
        static const int eeg_job_anc[] = { 6, 6, 6, 6, 6, 6, 6 };
        static const int eeg_job_rev[] = { 1, 1, 1, 1, 1, 1, 1 };
        static const int eeg_job_base[] = { 0, 3, 6, 9, 12, 15, 18 };
        static const T eeg_job_ax[] = { static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(1) };
        for(int job_idx = threadIdx.x + threadIdx.y*blockDim.x; job_idx < 7; job_idx += blockDim.x*blockDim.y){
            int j   = eeg_job_j[job_idx];
            int ee_anchor = eeg_job_anc[job_idx];
            int col_base = eeg_job_base[job_idx];
            T ax0 = eeg_job_ax[3*job_idx + 0]; T ax1 = eeg_job_ax[3*job_idx + 1]; T ax2 = eeg_job_ax[3*job_idx + 2];
            T axw_0 = s_Xworld[16*j + 0]*ax0 + s_Xworld[16*j + 4]*ax1 + s_Xworld[16*j + 8]*ax2;
            T axw_1 = s_Xworld[16*j + 1]*ax0 + s_Xworld[16*j + 5]*ax1 + s_Xworld[16*j + 9]*ax2;
            T axw_2 = s_Xworld[16*j + 2]*ax0 + s_Xworld[16*j + 6]*ax1 + s_Xworld[16*j + 10]*ax2;
            if (eeg_job_rev[job_idx]) {
                s_Jw[col_base + 0] = axw_0; s_Jw[col_base + 1] = axw_1; s_Jw[col_base + 2] = axw_2;
                T dx = s_Xworld[16*ee_anchor + 12] - s_Xworld[16*j + 12];
                T dy = s_Xworld[16*ee_anchor + 13] - s_Xworld[16*j + 13];
                T dz = s_Xworld[16*ee_anchor + 14] - s_Xworld[16*j + 14];
                s_Jv[col_base + 0] = axw_1*dz - axw_2*dy;
                s_Jv[col_base + 1] = axw_2*dx - axw_0*dz;
                s_Jv[col_base + 2] = axw_0*dy - axw_1*dx;
            }
            else {
                s_Jv[col_base + 0] = axw_0; s_Jv[col_base + 1] = axw_1; s_Jv[col_base + 2] = axw_2;
            }
        }
        __syncthreads();
        //
        // Step 4: extract (cy, sy, cp, sp) from each ee's world rotation for E(rpy)^{-1}
        //
        for(int ee = threadIdx.x + threadIdx.y*blockDim.x; ee < 1; ee += blockDim.x*blockDim.y){
            const int ee_jid = 6;
            T R20 = s_Xworld[16*ee_jid + 2];
            T R21 = s_Xworld[16*ee_jid + 6];
            T R22 = s_Xworld[16*ee_jid + 10];
            T R10 = s_Xworld[16*ee_jid + 1];
            T R00 = s_Xworld[16*ee_jid + 0];
            T cp_term = sqrt(R22*R22 + R21*R21);
            T yaw = atan2(R10, R00);
            T pitch = atan2(-R20, cp_term);
            s_E_sc[4*ee + 0] = cos(yaw);
            s_E_sc[4*ee + 1] = sin(yaw);
            s_E_sc[4*ee + 2] = cos(pitch);
            s_E_sc[4*ee + 3] = sin(pitch);
        }
        __syncthreads();
        //
        // Step 5: write s_end_effector_pose_gradient (rows 0..2 = J_v, rows 3..5 = E(rpy)^{-1} J_w)
        //
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 42; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int rem = ind / 6; int vi = rem % 7; int ee = rem / 7;
            int jv_base = 3 * (7 * ee + vi);
            if (row < 3) {
                s_end_effector_pose_gradient[ind] = s_Jv[jv_base + row];
            }
            else {
                T cy = s_E_sc[4*ee + 0]; T sy = s_E_sc[4*ee + 1]; T cp = s_E_sc[4*ee + 2]; T sp = s_E_sc[4*ee + 3];
                T Jw0 = s_Jw[jv_base + 0]; T Jw1 = s_Jw[jv_base + 1]; T Jw2 = s_Jw[jv_base + 2];
                T outv;
                if (row == 3) { outv = (cy*Jw0 + sy*Jw1) / cp; }
                else if (row == 4) { outv = -sy*Jw0 + cy*Jw1; }
                else { outv = (sp / cp) * (cy*Jw0 + sy*Jw1) + Jw2; }
                s_end_effector_pose_gradient[ind] = outv;
            }
        }
        __syncthreads();
    }

    /**
     * Computes the Gradient of the End Effector Pose with respect to joint position
     *
     * @param s_end_effector_pose_gradient is a pointer to shared memory of size 6*NUM_VEL*NUM_EE where NUM_VEL = 7 and NUM_EE = 1
     * @param s_q is the vector of joint positions
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     */
    template <typename T>
    __device__
    void end_effector_pose_gradient_device(T *s_end_effector_pose_gradient, const T *s_q, const robotModel<T> *d_robotModel) {
        // GRID shared arena layout
        //   T s_XmatsHom[144]
        //   T s_temp[190]
        //   bytes s_linalg_smem[GRID_EE_LINALG_SHARED_BYTES<T>()]
        extern __shared__ __align__(16) unsigned char s_arena[];
        size_t s_arena_offset = 0;
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_XmatsHom = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(144);
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_temp = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(190);
        int *s_topology_helpers = nullptr;
        unsigned char *s_linalg_smem = nullptr;
        if (static_cast<size_t>(GRID_EE_LINALG_SHARED_BYTES<T>()) > 0) {
            s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
            s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
            s_arena_offset += static_cast<size_t>(GRID_EE_LINALG_SHARED_BYTES<T>());
        }
        #ifdef GRID_CUDA_DEBUG_LAYOUT
        assert(s_arena_offset == grid_shared_arena_bytes<T>(334, 0, GRID_EE_LINALG_SHARED_BYTES<T>()));
        #endif
        (void)s_arena_offset;
        load_update_XmatsHom_helpers<T>(s_XmatsHom, s_topology_helpers, s_q, d_robotModel, s_temp);
        end_effector_pose_gradient_inner<T, true>(s_end_effector_pose_gradient, s_q, s_XmatsHom, nullptr, s_topology_helpers, s_temp, nullptr, s_linalg_smem);
    }

    /**
     * Computes the Gradient of the End Effector Pose with respect to joint position
     *
     * @param d_end_effector_pose_gradient is the vector of end effector positions gradients
     * @param d_workspace is the generated global spill workspace
     * @param d_q is the vector of joint positions
     * @param stride_q is the stide between each q
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER, bool MUJOCO_OUTPUT = false>
    __global__
    __launch_bounds__(tier_max_threads<RESOURCE_TIER>())
    void end_effector_pose_gradient_kernel_single_timing(T *d_end_effector_pose_gradient, unsigned char *d_workspace, const T *d_q, const int stride_q, const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS) {
        if constexpr (RESOURCE_TIER == TIER_SHARED) {
            // GRID shared arena layout
            //   T s_q[7]
            //   T s_end_effector_pose_gradient[42]
            //   T s_XmatsHom[144]
            //   T s_temp[190]
            //   bytes s_linalg_smem[GRID_EE_LINALG_SHARED_BYTES<T>()]
            extern __shared__ __align__(16) unsigned char s_arena[];
            size_t s_arena_offset = 0;
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_q = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(7);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_end_effector_pose_gradient = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(42);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_XmatsHom = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(144);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_temp = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(190);
            int *s_topology_helpers = nullptr;
            unsigned char *s_linalg_smem = nullptr;
            if (static_cast<size_t>(GRID_EE_LINALG_SHARED_BYTES<T>()) > 0) {
                s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
                s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
                s_arena_offset += static_cast<size_t>(GRID_EE_LINALG_SHARED_BYTES<T>());
            }
            #ifdef GRID_CUDA_DEBUG_LAYOUT
            assert(s_arena_offset == grid_shared_arena_bytes<T>(383, 0, GRID_EE_LINALG_SHARED_BYTES<T>()));
            #endif
            (void)s_arena_offset;
            (void)d_workspace;
            // load to shared mem
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                s_q[ind] = d_q[ind];
            }
            __syncthreads();
            // compute with NUM_TIMESTEPS as NUM_REPS for timing
            for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
                // anti-LICM: volatile reload of inputs each rep
                for(int _aopt_i = threadIdx.x + threadIdx.y*blockDim.x; _aopt_i < 7; _aopt_i += blockDim.x*blockDim.y){
                    reinterpret_cast<volatile T *>(s_q)[_aopt_i] = reinterpret_cast<const volatile T *>(d_q)[_aopt_i];
                }
                __syncthreads();
                // anti-LICM (1/2): stomp one input slot with `rep`
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
                    reinterpret_cast<volatile T *>(s_q)[rep % (7)] = static_cast<T>(rep);
                }
                // anti-LICM (2/2): feedback prev rep's d_end_effector_pose_gradient into s_q (true loop-carried dep)
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
                    T _aopt_fb1 = reinterpret_cast<const volatile T *>(d_end_effector_pose_gradient)[(rep + 0x3FF) & 0x3FF];
                    T _aopt_fb2 = reinterpret_cast<const volatile T *>(d_end_effector_pose_gradient)[(rep + 0x3FE) & 0x3FF];
                    T _aopt_fb3 = reinterpret_cast<const volatile T *>(d_end_effector_pose_gradient)[(rep + 0x3FD) & 0x3FF];
                    reinterpret_cast<volatile T *>(s_q)[(rep + 1) % (7)] += _aopt_fb1;
                    reinterpret_cast<volatile T *>(s_q)[(rep + 2) % (7)] += _aopt_fb2;
                    reinterpret_cast<volatile T *>(s_q)[(rep + 3) % (7)] += _aopt_fb3;
                }
                __syncthreads();
                load_update_XmatsHom_helpers<T>(s_XmatsHom, s_topology_helpers, s_q, d_robotModel, s_temp);
                end_effector_pose_gradient_inner<T, true>(s_end_effector_pose_gradient, s_q, s_XmatsHom, nullptr, s_topology_helpers, s_temp, nullptr, s_linalg_smem);
                __syncthreads();
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) { reinterpret_cast<volatile T *>(d_end_effector_pose_gradient)[rep & 1023] = reinterpret_cast<const volatile T *>(s_end_effector_pose_gradient)[rep & 7]; }
            }
            // save down to global
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 42; ind += blockDim.x*blockDim.y){
                d_end_effector_pose_gradient[ind] = s_end_effector_pose_gradient[ind];
            }
            __syncthreads();
        }
        else if constexpr (RESOURCE_TIER == TIER_LITE) {
            // GRID shared arena layout
            //   T s_q[7]
            //   T s_end_effector_pose_gradient[42]
            //   T s_XmatsHom[144]
            //   T s_temp[190]
            //   bytes s_linalg_smem[GRID_EE_LINALG_SHARED_BYTES<T>()]
            extern __shared__ __align__(16) unsigned char s_arena[];
            size_t s_arena_offset = 0;
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_q = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(7);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_end_effector_pose_gradient = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(42);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_XmatsHom = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(144);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_temp = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(190);
            int *s_topology_helpers = nullptr;
            unsigned char *s_linalg_smem = nullptr;
            if (static_cast<size_t>(GRID_EE_LINALG_SHARED_BYTES<T>()) > 0) {
                s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
                s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
                s_arena_offset += static_cast<size_t>(GRID_EE_LINALG_SHARED_BYTES<T>());
            }
            #ifdef GRID_CUDA_DEBUG_LAYOUT
            assert(s_arena_offset == grid_shared_arena_bytes<T>(383, 0, GRID_EE_LINALG_SHARED_BYTES<T>()));
            #endif
            (void)s_arena_offset;
            (void)d_workspace;
            // load to shared mem
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                s_q[ind] = d_q[ind];
            }
            __syncthreads();
            // compute with NUM_TIMESTEPS as NUM_REPS for timing
            for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
                // anti-LICM: volatile reload of inputs each rep
                for(int _aopt_i = threadIdx.x + threadIdx.y*blockDim.x; _aopt_i < 7; _aopt_i += blockDim.x*blockDim.y){
                    reinterpret_cast<volatile T *>(s_q)[_aopt_i] = reinterpret_cast<const volatile T *>(d_q)[_aopt_i];
                }
                __syncthreads();
                // anti-LICM (1/2): stomp one input slot with `rep`
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
                    reinterpret_cast<volatile T *>(s_q)[rep % (7)] = static_cast<T>(rep);
                }
                // anti-LICM (2/2): feedback prev rep's d_end_effector_pose_gradient into s_q (true loop-carried dep)
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
                    T _aopt_fb1 = reinterpret_cast<const volatile T *>(d_end_effector_pose_gradient)[(rep + 0x3FF) & 0x3FF];
                    T _aopt_fb2 = reinterpret_cast<const volatile T *>(d_end_effector_pose_gradient)[(rep + 0x3FE) & 0x3FF];
                    T _aopt_fb3 = reinterpret_cast<const volatile T *>(d_end_effector_pose_gradient)[(rep + 0x3FD) & 0x3FF];
                    reinterpret_cast<volatile T *>(s_q)[(rep + 1) % (7)] += _aopt_fb1;
                    reinterpret_cast<volatile T *>(s_q)[(rep + 2) % (7)] += _aopt_fb2;
                    reinterpret_cast<volatile T *>(s_q)[(rep + 3) % (7)] += _aopt_fb3;
                }
                __syncthreads();
                load_update_XmatsHom_helpers<T>(s_XmatsHom, s_topology_helpers, s_q, d_robotModel, s_temp);
                end_effector_pose_gradient_inner<T, true>(s_end_effector_pose_gradient, s_q, s_XmatsHom, nullptr, s_topology_helpers, s_temp, nullptr, s_linalg_smem);
                __syncthreads();
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) { reinterpret_cast<volatile T *>(d_end_effector_pose_gradient)[rep & 1023] = reinterpret_cast<const volatile T *>(s_end_effector_pose_gradient)[rep & 7]; }
            }
            // save down to global
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 42; ind += blockDim.x*blockDim.y){
                d_end_effector_pose_gradient[ind] = s_end_effector_pose_gradient[ind];
            }
            __syncthreads();
        }
        else if constexpr (RESOURCE_TIER == TIER_MINIMAL) {
            // GRID shared arena layout
            //   T s_q[7]
            //   T s_XmatsHom[144]
            //   bytes s_linalg_smem[GRID_EE_LINALG_SHARED_BYTES<T>()]
            extern __shared__ __align__(16) unsigned char s_arena[];
            size_t s_arena_offset = 0;
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_q = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(7);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_XmatsHom = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(144);
            T *s_temp = nullptr;
            int *s_topology_helpers = nullptr;
            unsigned char *s_linalg_smem = nullptr;
            if (static_cast<size_t>(GRID_EE_LINALG_SHARED_BYTES<T>()) > 0) {
                s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
                s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
                s_arena_offset += static_cast<size_t>(GRID_EE_LINALG_SHARED_BYTES<T>());
            }
            #ifdef GRID_CUDA_DEBUG_LAYOUT
            assert(s_arena_offset == grid_shared_arena_bytes<T>(151, 0, GRID_EE_LINALG_SHARED_BYTES<T>()));
            #endif
            (void)s_arena_offset;
            // load to shared mem
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                s_q[ind] = d_q[ind];
            }
            __syncthreads();
            T *s_end_effector_pose_gradient = d_end_effector_pose_gradient;
            T *s_eegrad_temp = reinterpret_cast<T *>(&d_workspace[GRID_END_EFFECTOR_POSE_GRADIENT_WORKSPACE_DXHOM_OFFSET_BYTES<T>()]);
            s_temp = s_eegrad_temp;
            // compute with NUM_TIMESTEPS as NUM_REPS for timing
            for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
                // anti-LICM: volatile reload of inputs each rep
                for(int _aopt_i = threadIdx.x + threadIdx.y*blockDim.x; _aopt_i < 7; _aopt_i += blockDim.x*blockDim.y){
                    reinterpret_cast<volatile T *>(s_q)[_aopt_i] = reinterpret_cast<const volatile T *>(d_q)[_aopt_i];
                }
                __syncthreads();
                // anti-LICM (1/2): stomp one input slot with `rep`
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
                    reinterpret_cast<volatile T *>(s_q)[rep % (7)] = static_cast<T>(rep);
                }
                // anti-LICM (2/2): feedback prev rep's d_end_effector_pose_gradient into s_q (true loop-carried dep)
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
                    T _aopt_fb1 = reinterpret_cast<const volatile T *>(d_end_effector_pose_gradient)[(rep + 0x3FF) & 0x3FF];
                    T _aopt_fb2 = reinterpret_cast<const volatile T *>(d_end_effector_pose_gradient)[(rep + 0x3FE) & 0x3FF];
                    T _aopt_fb3 = reinterpret_cast<const volatile T *>(d_end_effector_pose_gradient)[(rep + 0x3FD) & 0x3FF];
                    reinterpret_cast<volatile T *>(s_q)[(rep + 1) % (7)] += _aopt_fb1;
                    reinterpret_cast<volatile T *>(s_q)[(rep + 2) % (7)] += _aopt_fb2;
                    reinterpret_cast<volatile T *>(s_q)[(rep + 3) % (7)] += _aopt_fb3;
                }
                __syncthreads();
                load_update_XmatsHom_helpers<T>(s_XmatsHom, s_topology_helpers, s_q, d_robotModel, s_temp);
                end_effector_pose_gradient_inner<T, false>(s_end_effector_pose_gradient, s_q, s_XmatsHom, nullptr, s_topology_helpers, s_temp, s_eegrad_temp, s_linalg_smem);
                __syncthreads();
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) { reinterpret_cast<volatile T *>(d_end_effector_pose_gradient)[rep & 1023] = reinterpret_cast<const volatile T *>(s_end_effector_pose_gradient)[rep & 7]; }
            }
        }
    }

    /**
     * Computes the Gradient of the End Effector Pose with respect to joint position
     *
     * @param d_end_effector_pose_gradient is the vector of end effector positions gradients
     * @param d_workspace is the generated global spill workspace
     * @param d_q is the vector of joint positions
     * @param stride_q is the stide between each q
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER, bool MUJOCO_OUTPUT = false>
    __global__
    __launch_bounds__(tier_max_threads<RESOURCE_TIER>())
    void end_effector_pose_gradient_kernel(T *d_end_effector_pose_gradient, unsigned char *d_workspace, const T *d_q, const int stride_q, const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS) {
        if constexpr (RESOURCE_TIER == TIER_SHARED) {
            // GRID shared arena layout
            //   T s_q[7]
            //   T s_end_effector_pose_gradient[42]
            //   T s_XmatsHom[144]
            //   T s_temp[190]
            //   bytes s_linalg_smem[GRID_EE_LINALG_SHARED_BYTES<T>()]
            extern __shared__ __align__(16) unsigned char s_arena[];
            size_t s_arena_offset = 0;
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_q = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(7);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_end_effector_pose_gradient = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(42);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_XmatsHom = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(144);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_temp = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(190);
            int *s_topology_helpers = nullptr;
            unsigned char *s_linalg_smem = nullptr;
            if (static_cast<size_t>(GRID_EE_LINALG_SHARED_BYTES<T>()) > 0) {
                s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
                s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
                s_arena_offset += static_cast<size_t>(GRID_EE_LINALG_SHARED_BYTES<T>());
            }
            #ifdef GRID_CUDA_DEBUG_LAYOUT
            assert(s_arena_offset == grid_shared_arena_bytes<T>(383, 0, GRID_EE_LINALG_SHARED_BYTES<T>()));
            #endif
            (void)s_arena_offset;
            (void)d_workspace;
            for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
                // load to shared mem
                const T *d_q_k = &d_q[k*stride_q];
                for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                    s_q[ind] = d_q_k[ind];
                }
                __syncthreads();
                // compute
                load_update_XmatsHom_helpers<T>(s_XmatsHom, s_topology_helpers, s_q, d_robotModel, s_temp);
                end_effector_pose_gradient_inner<T, true>(s_end_effector_pose_gradient, s_q, s_XmatsHom, nullptr, s_topology_helpers, s_temp, nullptr, s_linalg_smem);
                __syncthreads();
                // save down to global
                T *d_end_effector_pose_gradient_k = &d_end_effector_pose_gradient[k*42];
                for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 42; ind += blockDim.x*blockDim.y){
                    d_end_effector_pose_gradient_k[ind] = s_end_effector_pose_gradient[ind];
                }
                __syncthreads();
            }
        }
        else if constexpr (RESOURCE_TIER == TIER_LITE) {
            // GRID shared arena layout
            //   T s_q[7]
            //   T s_end_effector_pose_gradient[42]
            //   T s_XmatsHom[144]
            //   T s_temp[190]
            //   bytes s_linalg_smem[GRID_EE_LINALG_SHARED_BYTES<T>()]
            extern __shared__ __align__(16) unsigned char s_arena[];
            size_t s_arena_offset = 0;
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_q = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(7);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_end_effector_pose_gradient = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(42);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_XmatsHom = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(144);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_temp = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(190);
            int *s_topology_helpers = nullptr;
            unsigned char *s_linalg_smem = nullptr;
            if (static_cast<size_t>(GRID_EE_LINALG_SHARED_BYTES<T>()) > 0) {
                s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
                s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
                s_arena_offset += static_cast<size_t>(GRID_EE_LINALG_SHARED_BYTES<T>());
            }
            #ifdef GRID_CUDA_DEBUG_LAYOUT
            assert(s_arena_offset == grid_shared_arena_bytes<T>(383, 0, GRID_EE_LINALG_SHARED_BYTES<T>()));
            #endif
            (void)s_arena_offset;
            (void)d_workspace;
            for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
                // load to shared mem
                const T *d_q_k = &d_q[k*stride_q];
                for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                    s_q[ind] = d_q_k[ind];
                }
                __syncthreads();
                // compute
                load_update_XmatsHom_helpers<T>(s_XmatsHom, s_topology_helpers, s_q, d_robotModel, s_temp);
                end_effector_pose_gradient_inner<T, true>(s_end_effector_pose_gradient, s_q, s_XmatsHom, nullptr, s_topology_helpers, s_temp, nullptr, s_linalg_smem);
                __syncthreads();
                // save down to global
                T *d_end_effector_pose_gradient_k = &d_end_effector_pose_gradient[k*42];
                for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 42; ind += blockDim.x*blockDim.y){
                    d_end_effector_pose_gradient_k[ind] = s_end_effector_pose_gradient[ind];
                }
                __syncthreads();
            }
        }
        else if constexpr (RESOURCE_TIER == TIER_MINIMAL) {
            // GRID shared arena layout
            //   T s_q[7]
            //   T s_XmatsHom[144]
            //   bytes s_linalg_smem[GRID_EE_LINALG_SHARED_BYTES<T>()]
            extern __shared__ __align__(16) unsigned char s_arena[];
            size_t s_arena_offset = 0;
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_q = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(7);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_XmatsHom = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(144);
            T *s_temp = nullptr;
            int *s_topology_helpers = nullptr;
            unsigned char *s_linalg_smem = nullptr;
            if (static_cast<size_t>(GRID_EE_LINALG_SHARED_BYTES<T>()) > 0) {
                s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
                s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
                s_arena_offset += static_cast<size_t>(GRID_EE_LINALG_SHARED_BYTES<T>());
            }
            #ifdef GRID_CUDA_DEBUG_LAYOUT
            assert(s_arena_offset == grid_shared_arena_bytes<T>(151, 0, GRID_EE_LINALG_SHARED_BYTES<T>()));
            #endif
            (void)s_arena_offset;
            for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
                // load to shared mem
                const T *d_q_k = &d_q[k*stride_q];
                for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                    s_q[ind] = d_q_k[ind];
                }
                __syncthreads();
                T *s_end_effector_pose_gradient = &d_end_effector_pose_gradient[k*42];
                T *s_eegrad_temp = reinterpret_cast<T *>(&d_workspace[grid_workspace_slot()*GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>() + GRID_END_EFFECTOR_POSE_GRADIENT_WORKSPACE_DXHOM_OFFSET_BYTES<T>()]);
                s_temp = s_eegrad_temp;
                // compute
                load_update_XmatsHom_helpers<T>(s_XmatsHom, s_topology_helpers, s_q, d_robotModel, s_temp);
                end_effector_pose_gradient_inner<T, false>(s_end_effector_pose_gradient, s_q, s_XmatsHom, nullptr, s_topology_helpers, s_temp, s_eegrad_temp, s_linalg_smem);
                __syncthreads();
            }
        }
    }

    /**
     * Computes the Gradient of the End Effector Pose with respect to joint position
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_COMPRESSED_MEM = false, gridDataKind KIND = GRID_DATA_ALL, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>
    __host__
    void end_effector_pose_gradient(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps,
                                const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_KINEMATICS, "end_effector_pose_gradient requires all-data or kinematics gridData");
        // start code with memory transfer
        int stride_q;
        if (USE_COMPRESSED_MEM) {stride_q = NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q,hd_data->h_q,stride_q*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        else {stride_q = 3*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        gpuErrchkKernel();
        // then call the kernel
        gpuErrchk(grid_check_dynamic_shared_memory_bytes("end_effector_pose_gradient", END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()));
        const int _grid_ws_n = (hd_data->workspace_timestep_slots > 0 && hd_data->workspace_timestep_slots < num_timesteps) ? hd_data->workspace_timestep_slots : num_timesteps;
        if (GRID_END_EFFECTOR_POSE_GRADIENT_USES_WORKSPACE_TEMP_ANY) {gpuErrchk(grid_begin_l2_persisting(0, hd_data->d_workspace, GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()*static_cast<size_t>(_grid_ws_n)));}
        dim3 _ws_grid = block_dimms;
        if ((int)(_ws_grid.x*_ws_grid.y*_ws_grid.z) > _grid_ws_n) { _ws_grid = dim3(_grid_ws_n,1,1); }
        dim3 _grid_thr_clamped_4 = grid_host_clamp_threads((const void*)&end_effector_pose_gradient_kernel<T, RESOURCE_TIER>, thread_dimms);
        if (USE_COMPRESSED_MEM) {end_effector_pose_gradient_kernel<T, RESOURCE_TIER><<<_ws_grid,_grid_thr_clamped_4,END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()>>>(hd_data->d_end_effector_pose_gradient,hd_data->d_workspace,hd_data->d_q,stride_q,d_robotModel,num_timesteps);}
        else                    {end_effector_pose_gradient_kernel<T, RESOURCE_TIER><<<_ws_grid,_grid_thr_clamped_4,END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()>>>(hd_data->d_end_effector_pose_gradient,hd_data->d_workspace,hd_data->d_q_qd_u,stride_q,d_robotModel,num_timesteps);}
        gpuErrchkKernel();
        if (GRID_END_EFFECTOR_POSE_GRADIENT_USES_WORKSPACE_TEMP_ANY) {gpuErrchk(grid_end_l2_persisting(0));}
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_end_effector_pose_gradient,hd_data->d_end_effector_pose_gradient,6*NUM_EES*NUM_VEL*num_timesteps*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchkKernel();
    }

    /**
     * Computes the Gradient of the End Effector Pose with respect to joint position
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_COMPRESSED_MEM = false, gridDataKind KIND = GRID_DATA_ALL, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>
    __host__
    void end_effector_pose_gradient_single_timing(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps,
                                              const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_KINEMATICS, "end_effector_pose_gradient requires all-data or kinematics gridData");
        // start code with memory transfer
        int stride_q;
        if (USE_COMPRESSED_MEM) {stride_q = NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q,hd_data->h_q,stride_q*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        else {stride_q = 3*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        gpuErrchkKernel();
        // then call the kernel
        gpuErrchk(grid_check_dynamic_shared_memory_bytes("end_effector_pose_gradient", END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()));
        if (GRID_END_EFFECTOR_POSE_GRADIENT_USES_WORKSPACE_TEMP_ANY) {gpuErrchk(grid_begin_l2_persisting(0, hd_data->d_workspace, GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()));}
        struct timespec start, end; clock_gettime(CLOCK_MONOTONIC,&start);
        dim3 _grid_thr_clamped_5 = grid_host_clamp_threads((const void*)&end_effector_pose_gradient_kernel_single_timing<T, RESOURCE_TIER>, thread_dimms);
        if (USE_COMPRESSED_MEM) {end_effector_pose_gradient_kernel_single_timing<T, RESOURCE_TIER><<<block_dimms,_grid_thr_clamped_5,END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()>>>(hd_data->d_end_effector_pose_gradient,hd_data->d_workspace,hd_data->d_q,stride_q,d_robotModel,num_timesteps);}
        else                    {end_effector_pose_gradient_kernel_single_timing<T, RESOURCE_TIER><<<block_dimms,_grid_thr_clamped_5,END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()>>>(hd_data->d_end_effector_pose_gradient,hd_data->d_workspace,hd_data->d_q_qd_u,stride_q,d_robotModel,num_timesteps);}
        gpuErrchkKernel();
        clock_gettime(CLOCK_MONOTONIC,&end);
        if (GRID_END_EFFECTOR_POSE_GRADIENT_USES_WORKSPACE_TEMP_ANY) {gpuErrchk(grid_end_l2_persisting(0));}
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_end_effector_pose_gradient,hd_data->d_end_effector_pose_gradient,6*NUM_EES*NUM_VEL*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchkKernel();
        printf("Single Call END_EFFECTOR_POSE_GRADIENT %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(num_timesteps));
    }

    /**
     * Computes the Gradient of the End Effector Pose with respect to joint position
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_COMPRESSED_MEM = false, gridDataKind KIND = GRID_DATA_ALL, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>
    __host__
    void end_effector_pose_gradient_compute_only(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps,
                                             const dim3 block_dimms, const dim3 thread_dimms) {
        static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_KINEMATICS, "end_effector_pose_gradient requires all-data or kinematics gridData");
        int stride_q = USE_COMPRESSED_MEM ? NUM_JOINTS: 3*NUM_JOINTS;
        // then call the kernel
        gpuErrchk(grid_check_dynamic_shared_memory_bytes("end_effector_pose_gradient", END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()));
        const int _grid_ws_n = (hd_data->workspace_timestep_slots > 0 && hd_data->workspace_timestep_slots < num_timesteps) ? hd_data->workspace_timestep_slots : num_timesteps;
        if (GRID_END_EFFECTOR_POSE_GRADIENT_USES_WORKSPACE_TEMP_ANY) {gpuErrchk(grid_begin_l2_persisting(0, hd_data->d_workspace, GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()*static_cast<size_t>(_grid_ws_n)));}
        dim3 _ws_grid = block_dimms;
        if ((int)(_ws_grid.x*_ws_grid.y*_ws_grid.z) > _grid_ws_n) { _ws_grid = dim3(_grid_ws_n,1,1); }
        dim3 _grid_thr_clamped_6 = grid_host_clamp_threads((const void*)&end_effector_pose_gradient_kernel<T, RESOURCE_TIER>, thread_dimms);
        if (USE_COMPRESSED_MEM) {end_effector_pose_gradient_kernel<T, RESOURCE_TIER><<<_ws_grid,_grid_thr_clamped_6,END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()>>>(hd_data->d_end_effector_pose_gradient,hd_data->d_workspace,hd_data->d_q,stride_q,d_robotModel,num_timesteps);}
        else                    {end_effector_pose_gradient_kernel<T, RESOURCE_TIER><<<_ws_grid,_grid_thr_clamped_6,END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()>>>(hd_data->d_end_effector_pose_gradient,hd_data->d_workspace,hd_data->d_q_qd_u,stride_q,d_robotModel,num_timesteps);}
        gpuErrchkKernel();
        if (GRID_END_EFFECTOR_POSE_GRADIENT_USES_WORKSPACE_TEMP_ANY) {gpuErrchk(grid_end_l2_persisting(0));}
    }

    /**
     * Computes the End Effector Position
     *
     * Notes:
     *   Assumes the Xhom matricies have already been updated for the given q
     *   Defaults to all leave nodes if fixed_target_name is not provided
     *
     * @param s_end_effector_pose is a pointer to shared memory of size 6*NUM_EE where NUM_EE = 1
     * @param s_q is the vector of joint positions
     * @param s_Xhom is the pointer to the homogenous transformation matricies 
     * @param s_temp is a pointer to helper shared memory of size 32
     * @param d_workspace is the global-memory chain workspace used in place of s_temp when !TEMP_IN_SMEM
     * @param s_topology_helpers is the (shared) memory location for the topology_helpers (nullptr/unused for serial chains with identical Ss)
     * @param s_linalg_smem is optional byte-addressed shared memory (reserved; unused by this inner)
     */
    template <typename T, bool TEMP_IN_SMEM = true>
    __device__
    void end_effector_pose_inner_EE(T *s_end_effector_pose, const T *s_q, const T *s_Xhom, int *s_topology_helpers, T *s_temp, T *d_workspace, unsigned char *s_linalg_smem) {
        if constexpr (!TEMP_IN_SMEM) { s_temp = d_workspace; } else { (void)d_workspace; }
        //
        // For each branch in parallel chain up the transform
        // Keep chaining until reaching the root (starting from the leaves)
        //
        // Serial chain manipulator so optimize as parent is jid-1
        // First set to leaf (or fixed) transform
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 16; ind += blockDim.x*blockDim.y){
            s_temp[ind] = s_Xhom[16*7 + ind];
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 1/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 16; ind += blockDim.x*blockDim.y){
            int row = ind % 4; int col = ind / 4;
            s_temp[ind + 16] = dot_prod<T,4,4,1>(&s_Xhom[16*6 + row], &s_temp[0 + 4*col]);
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 2/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 16; ind += blockDim.x*blockDim.y){
            int row = ind % 4; int col = ind / 4;
            s_temp[ind + 0] = dot_prod<T,4,4,1>(&s_Xhom[16*5 + row], &s_temp[16 + 4*col]);
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 3/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 16; ind += blockDim.x*blockDim.y){
            int row = ind % 4; int col = ind / 4;
            s_temp[ind + 16] = dot_prod<T,4,4,1>(&s_Xhom[16*4 + row], &s_temp[0 + 4*col]);
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 4/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 16; ind += blockDim.x*blockDim.y){
            int row = ind % 4; int col = ind / 4;
            s_temp[ind + 0] = dot_prod<T,4,4,1>(&s_Xhom[16*3 + row], &s_temp[16 + 4*col]);
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 5/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 16; ind += blockDim.x*blockDim.y){
            int row = ind % 4; int col = ind / 4;
            s_temp[ind + 16] = dot_prod<T,4,4,1>(&s_Xhom[16*2 + row], &s_temp[0 + 4*col]);
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 6/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 16; ind += blockDim.x*blockDim.y){
            int row = ind % 4; int col = ind / 4;
            s_temp[ind + 0] = dot_prod<T,4,4,1>(&s_Xhom[16*1 + row], &s_temp[16 + 4*col]);
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 7/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 16; ind += blockDim.x*blockDim.y){
            int row = ind % 4; int col = ind / 4;
            s_temp[ind + 16] = dot_prod<T,4,4,1>(&s_Xhom[16*0 + row], &s_temp[0 + 4*col]);
        }
        __syncthreads();
        //
        // Now extract the end_effector_pose from the transforms.
        // (This generic family evaluates the last MOVING joint; a terminal fixed
        // joint's <origin> is tracked by the named fixed-target family instead.)
        //
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 3; ind += blockDim.x*blockDim.y){
            // xyz is easy
            int xyzInd = ind % 3; int eeInd = ind / 3; T *s_Xmat_hom = &s_temp[16 + 16*eeInd];
            s_end_effector_pose[6*eeInd + xyzInd] = s_Xmat_hom[12 + xyzInd];
            // roll pitch yaw is a bit more difficult
            if(xyzInd > 0){continue;}
            s_end_effector_pose[6*eeInd + 3] = atan2(s_Xmat_hom[6],s_Xmat_hom[10]);
            s_end_effector_pose[6*eeInd + 4] = -atan2(s_Xmat_hom[2],sqrt(s_Xmat_hom[6]*s_Xmat_hom[6] + s_Xmat_hom[10]*s_Xmat_hom[10]));
            s_end_effector_pose[6*eeInd + 5] = atan2(s_Xmat_hom[1],s_Xmat_hom[0]);
        }
        __syncthreads();
    }

    /**
     * Computes the End Effector Position
     *
     * @param s_end_effector_pose is a pointer to shared memory of size 6*NUM_EE where NUM_EE = 1
     * @param s_q is the vector of joint positions
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     */
    template <typename T>
    __device__
    void end_effector_pose_device_EE(T *s_end_effector_pose, const T *s_q, const robotModel<T> *d_robotModel) {
        // GRID shared arena layout
        //   T s_XmatsHom[144]
        //   T s_temp[32]
        //   bytes s_linalg_smem[GRID_EE_LINALG_SHARED_BYTES<T>()]
        extern __shared__ __align__(16) unsigned char s_arena[];
        size_t s_arena_offset = 0;
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_XmatsHom = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(144);
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_temp = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(32);
        int *s_topology_helpers = nullptr;
        unsigned char *s_linalg_smem = nullptr;
        if (static_cast<size_t>(GRID_EE_LINALG_SHARED_BYTES<T>()) > 0) {
            s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
            s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
            s_arena_offset += static_cast<size_t>(GRID_EE_LINALG_SHARED_BYTES<T>());
        }
        #ifdef GRID_CUDA_DEBUG_LAYOUT
        assert(s_arena_offset == grid_shared_arena_bytes<T>(176, 0, GRID_EE_LINALG_SHARED_BYTES<T>()));
        #endif
        (void)s_arena_offset;
        load_update_XmatsHom_helpers<T>(s_XmatsHom, s_topology_helpers, s_q, d_robotModel, s_temp);
        end_effector_pose_inner_EE<T, true>(s_end_effector_pose, s_q, s_XmatsHom, s_topology_helpers, s_temp, nullptr, s_linalg_smem);
    }

    /**
     * Compute the End Effector Position
     *
     * @param d_end_effector_pose is the vector of end effector positions
     * @param d_q is the vector of joint positions
     * @param stride_q is the stide between each q
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER, bool MUJOCO_OUTPUT = false>
    __global__
    __launch_bounds__(tier_max_threads<RESOURCE_TIER>())
    void end_effector_pose_kernel_EE_single_timing(T *d_end_effector_pose, const T *d_q, const int stride_q, const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS) {
        // GRID shared arena layout
        //   T s_q[7]
        //   T s_end_effector_pose[6]
        //   T s_XmatsHom[144]
        //   T s_temp[32]
        //   bytes s_linalg_smem[GRID_EE_LINALG_SHARED_BYTES<T>()]
        extern __shared__ __align__(16) unsigned char s_arena[];
        size_t s_arena_offset = 0;
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_q = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(7);
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_end_effector_pose = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(6);
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_XmatsHom = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(144);
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_temp = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(32);
        int *s_topology_helpers = nullptr;
        unsigned char *s_linalg_smem = nullptr;
        if (static_cast<size_t>(GRID_EE_LINALG_SHARED_BYTES<T>()) > 0) {
            s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
            s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
            s_arena_offset += static_cast<size_t>(GRID_EE_LINALG_SHARED_BYTES<T>());
        }
        #ifdef GRID_CUDA_DEBUG_LAYOUT
        assert(s_arena_offset == grid_shared_arena_bytes<T>(189, 0, GRID_EE_LINALG_SHARED_BYTES<T>()));
        #endif
        (void)s_arena_offset;
        // load to shared mem
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
            s_q[ind] = d_q[ind];
        }
        __syncthreads();
        // compute with NUM_TIMESTEPS as NUM_REPS for timing
        for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
            // anti-LICM: volatile reload of inputs each rep
            for(int _aopt_i = threadIdx.x + threadIdx.y*blockDim.x; _aopt_i < 7; _aopt_i += blockDim.x*blockDim.y){
                reinterpret_cast<volatile T *>(s_q)[_aopt_i] = reinterpret_cast<const volatile T *>(d_q)[_aopt_i];
            }
            __syncthreads();
            // anti-LICM (1/2): stomp one input slot with `rep`
            if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
                reinterpret_cast<volatile T *>(s_q)[rep % (7)] = static_cast<T>(rep);
            }
            // anti-LICM (2/2): feedback prev rep's d_end_effector_pose into s_q (true loop-carried dep)
            if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
                T _aopt_fb1 = reinterpret_cast<const volatile T *>(d_end_effector_pose)[(rep + 0x3FF) & 0x3FF];
                T _aopt_fb2 = reinterpret_cast<const volatile T *>(d_end_effector_pose)[(rep + 0x3FE) & 0x3FF];
                T _aopt_fb3 = reinterpret_cast<const volatile T *>(d_end_effector_pose)[(rep + 0x3FD) & 0x3FF];
                reinterpret_cast<volatile T *>(s_q)[(rep + 1) % (7)] += _aopt_fb1;
                reinterpret_cast<volatile T *>(s_q)[(rep + 2) % (7)] += _aopt_fb2;
                reinterpret_cast<volatile T *>(s_q)[(rep + 3) % (7)] += _aopt_fb3;
            }
            __syncthreads();
            load_update_XmatsHom_helpers<T>(s_XmatsHom, s_topology_helpers, s_q, d_robotModel, s_temp);
            end_effector_pose_inner_EE<T, true>(s_end_effector_pose, s_q, s_XmatsHom, s_topology_helpers, s_temp, nullptr, s_linalg_smem);
            __syncthreads();
            if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) { reinterpret_cast<volatile T *>(d_end_effector_pose)[rep & 1023] = reinterpret_cast<const volatile T *>(s_end_effector_pose)[rep & 7]; }
        }
        // save down to global
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            d_end_effector_pose[ind] = s_end_effector_pose[ind];
        }
        __syncthreads();
    }

    /**
     * Compute the End Effector Position
     *
     * @param d_end_effector_pose is the vector of end effector positions
     * @param d_q is the vector of joint positions
     * @param stride_q is the stide between each q
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER, bool MUJOCO_OUTPUT = false>
    __global__
    __launch_bounds__(tier_max_threads<RESOURCE_TIER>())
    void end_effector_pose_kernel_EE(T *d_end_effector_pose, const T *d_q, const int stride_q, const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS) {
        // GRID shared arena layout
        //   T s_q[7]
        //   T s_end_effector_pose[6]
        //   T s_XmatsHom[144]
        //   T s_temp[32]
        //   bytes s_linalg_smem[GRID_EE_LINALG_SHARED_BYTES<T>()]
        extern __shared__ __align__(16) unsigned char s_arena[];
        size_t s_arena_offset = 0;
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_q = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(7);
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_end_effector_pose = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(6);
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_XmatsHom = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(144);
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_temp = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(32);
        int *s_topology_helpers = nullptr;
        unsigned char *s_linalg_smem = nullptr;
        if (static_cast<size_t>(GRID_EE_LINALG_SHARED_BYTES<T>()) > 0) {
            s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
            s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
            s_arena_offset += static_cast<size_t>(GRID_EE_LINALG_SHARED_BYTES<T>());
        }
        #ifdef GRID_CUDA_DEBUG_LAYOUT
        assert(s_arena_offset == grid_shared_arena_bytes<T>(189, 0, GRID_EE_LINALG_SHARED_BYTES<T>()));
        #endif
        (void)s_arena_offset;
        for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
            // load to shared mem
            const T *d_q_k = &d_q[k*stride_q];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                s_q[ind] = d_q_k[ind];
            }
            __syncthreads();
            // compute
            load_update_XmatsHom_helpers<T>(s_XmatsHom, s_topology_helpers, s_q, d_robotModel, s_temp);
            end_effector_pose_inner_EE<T, true>(s_end_effector_pose, s_q, s_XmatsHom, s_topology_helpers, s_temp, nullptr, s_linalg_smem);
            __syncthreads();
            // save down to global
            T *d_end_effector_pose_k = &d_end_effector_pose[k*6];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
                d_end_effector_pose_k[ind] = s_end_effector_pose[ind];
            }
            __syncthreads();
        }
    }

    /**
     * Compute the End Effector Pose
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_COMPRESSED_MEM = false, gridDataKind KIND = GRID_DATA_ALL, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>
    __host__
    void end_effector_pose_EE(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps,
                                const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_KINEMATICS, "end_effector_pose requires all-data or kinematics gridData");
        // start code with memory transfer
        int stride_q;
        if (USE_COMPRESSED_MEM) {stride_q = NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q,hd_data->h_q,stride_q*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        else {stride_q = 3*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        gpuErrchkKernel();
        // then call the kernel
        gpuErrchk(grid_check_dynamic_shared_memory_bytes("end_effector_pose", END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_BYTES<T>()));
        dim3 _grid_thr_clamped_7 = grid_host_clamp_threads((const void*)&end_effector_pose_kernel_EE<T, RESOURCE_TIER>, thread_dimms);
        if (USE_COMPRESSED_MEM) {end_effector_pose_kernel_EE<T, RESOURCE_TIER><<<block_dimms,_grid_thr_clamped_7,END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_BYTES<T>()>>>(hd_data->d_end_effector_pose,hd_data->d_q,stride_q,d_robotModel,num_timesteps);}
        else                    {end_effector_pose_kernel_EE<T, RESOURCE_TIER><<<block_dimms,_grid_thr_clamped_7,END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_BYTES<T>()>>>(hd_data->d_end_effector_pose,hd_data->d_q_qd_u,stride_q,d_robotModel,num_timesteps);}
        gpuErrchkKernel();
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_end_effector_pose,hd_data->d_end_effector_pose,6*NUM_EES*num_timesteps*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchkKernel();
    }

    /**
     * Compute the End Effector Pose
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_COMPRESSED_MEM = false, gridDataKind KIND = GRID_DATA_ALL, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>
    __host__
    void end_effector_pose_EE_single_timing(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps,
                                              const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_KINEMATICS, "end_effector_pose requires all-data or kinematics gridData");
        // start code with memory transfer
        int stride_q;
        if (USE_COMPRESSED_MEM) {stride_q = NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q,hd_data->h_q,stride_q*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        else {stride_q = 3*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        gpuErrchkKernel();
        // then call the kernel
        gpuErrchk(grid_check_dynamic_shared_memory_bytes("end_effector_pose", END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_BYTES<T>()));
        struct timespec start, end; clock_gettime(CLOCK_MONOTONIC,&start);
        dim3 _grid_thr_clamped_8 = grid_host_clamp_threads((const void*)&end_effector_pose_kernel_EE<T, RESOURCE_TIER>, thread_dimms);
        if (USE_COMPRESSED_MEM) {end_effector_pose_kernel_EE<T, RESOURCE_TIER><<<block_dimms,_grid_thr_clamped_8,END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_BYTES<T>()>>>(hd_data->d_end_effector_pose,hd_data->d_q,stride_q,d_robotModel,num_timesteps);}
        else                    {end_effector_pose_kernel_EE<T, RESOURCE_TIER><<<block_dimms,_grid_thr_clamped_8,END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_BYTES<T>()>>>(hd_data->d_end_effector_pose,hd_data->d_q_qd_u,stride_q,d_robotModel,num_timesteps);}
        gpuErrchkKernel();
        clock_gettime(CLOCK_MONOTONIC,&end);
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_end_effector_pose,hd_data->d_end_effector_pose,6*NUM_EES*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchkKernel();
        printf("Single Call END_EFFECTOR_POSE %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(num_timesteps));
    }

    /**
     * Compute the End Effector Pose
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_COMPRESSED_MEM = false, gridDataKind KIND = GRID_DATA_ALL, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>
    __host__
    void end_effector_pose_EE_compute_only(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps,
                                             const dim3 block_dimms, const dim3 thread_dimms) {
        static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_KINEMATICS, "end_effector_pose requires all-data or kinematics gridData");
        int stride_q = USE_COMPRESSED_MEM ? NUM_JOINTS: 3*NUM_JOINTS;
        // then call the kernel
        gpuErrchk(grid_check_dynamic_shared_memory_bytes("end_effector_pose", END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_BYTES<T>()));
        dim3 _grid_thr_clamped_9 = grid_host_clamp_threads((const void*)&end_effector_pose_kernel_EE<T, RESOURCE_TIER>, thread_dimms);
        if (USE_COMPRESSED_MEM) {end_effector_pose_kernel_EE<T, RESOURCE_TIER><<<block_dimms,_grid_thr_clamped_9,END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_BYTES<T>()>>>(hd_data->d_end_effector_pose,hd_data->d_q,stride_q,d_robotModel,num_timesteps);}
        else                    {end_effector_pose_kernel_EE<T, RESOURCE_TIER><<<block_dimms,_grid_thr_clamped_9,END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_BYTES<T>()>>>(hd_data->d_end_effector_pose,hd_data->d_q_qd_u,stride_q,d_robotModel,num_timesteps);}
        gpuErrchkKernel();
    }

    /**
     * Computes the Gradient of the End Effector Pose with respect to generalized velocity (d/dv tangent, pinocchio convention)
     *
     * Notes:
     *   Assumes s_Xhom has been populated with the per-joint LOCAL transforms for the given q.
     *   Output d/dv (TANGENT) is 6 x nv per ee (was 6 x nq for d/dq) -- matches pinocchio.
     *
     * @param s_end_effector_pose_gradient is a pointer to shared memory of size 6*NUM_VEL*NUM_EE where NUM_VEL = 7 and NUM_EE = 1
     * @param s_q is the vector of joint positions (unused; kept for signature compatibility)
     * @param s_Xhom is the pointer to the LOCAL homogeneous transformation matrices (per-joint Xhom_local)
     * @param s_dXhom is the pointer to the LOCAL d-transforms (unused by the geometric-Jacobian path; kept for signature compatibility)
     * @param s_temp is a pointer to helper shared memory of size 190
     * @param d_workspace is the global-memory chain workspace used in place of s_temp when !TEMP_IN_SMEM
     * @param s_topology_helpers is the (shared) memory location for the topology_helpers (nullptr/unused for serial chains with identical Ss)
     * @param s_linalg_smem is optional byte-addressed shared memory (reserved; unused)
     */
    template <typename T, bool TEMP_IN_SMEM = true>
    __device__
    void end_effector_pose_gradient_inner_EE(T *s_end_effector_pose_gradient, const T *s_q, const T *s_Xhom, const T *s_dXhom, int *s_topology_helpers, T *s_temp, T *d_workspace, unsigned char *s_linalg_smem) {
        if constexpr (!TEMP_IN_SMEM) { s_temp = d_workspace; } else { (void)d_workspace; }
        (void)s_q; (void)s_dXhom; (void)s_linalg_smem;
        // scratch layout: Xworld | Jv (3 x nv x ee) | Jw (3 x nv x ee) | E_sincos (4 x ee)
        T *s_Xworld = &s_temp[0];
        T *s_Jv     = &s_temp[144];
        T *s_Jw     = &s_temp[165];
        T *s_E_sc   = &s_temp[186];   // cy,sy,cp,sp per ee
        //
        // Step 1: build world transforms for every joint via BFS-level chain-up
        //
        // BFS level 0 -> joints [0]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 16; ind += blockDim.x*blockDim.y){
            int slot = ind / 16; int ele = ind % 16;
            int row = ele & 3; int col = ele >> 2;
            // branch to get pointer locations
            int jid; int par;
                 if (slot < 1){ jid = 0; par = -1; }
            if (par == -1) {
                s_Xworld[16*jid + ele] = s_Xhom[16*jid + ele];
            }
            else {
                s_Xworld[16*jid + ele] = dot_prod<T,4,4,1>(&s_Xworld[16*par + row], &s_Xhom[16*jid + 4*col]);
            }
        }
        __syncthreads();
        // BFS level 1 -> joints [1]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 16; ind += blockDim.x*blockDim.y){
            int slot = ind / 16; int ele = ind % 16;
            int row = ele & 3; int col = ele >> 2;
            // branch to get pointer locations
            int jid; int par;
                 if (slot < 1){ jid = 1; par = 0; }
            if (par == -1) {
                s_Xworld[16*jid + ele] = s_Xhom[16*jid + ele];
            }
            else {
                s_Xworld[16*jid + ele] = dot_prod<T,4,4,1>(&s_Xworld[16*par + row], &s_Xhom[16*jid + 4*col]);
            }
        }
        __syncthreads();
        // BFS level 2 -> joints [2]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 16; ind += blockDim.x*blockDim.y){
            int slot = ind / 16; int ele = ind % 16;
            int row = ele & 3; int col = ele >> 2;
            // branch to get pointer locations
            int jid; int par;
                 if (slot < 1){ jid = 2; par = 1; }
            if (par == -1) {
                s_Xworld[16*jid + ele] = s_Xhom[16*jid + ele];
            }
            else {
                s_Xworld[16*jid + ele] = dot_prod<T,4,4,1>(&s_Xworld[16*par + row], &s_Xhom[16*jid + 4*col]);
            }
        }
        __syncthreads();
        // BFS level 3 -> joints [3]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 16; ind += blockDim.x*blockDim.y){
            int slot = ind / 16; int ele = ind % 16;
            int row = ele & 3; int col = ele >> 2;
            // branch to get pointer locations
            int jid; int par;
                 if (slot < 1){ jid = 3; par = 2; }
            if (par == -1) {
                s_Xworld[16*jid + ele] = s_Xhom[16*jid + ele];
            }
            else {
                s_Xworld[16*jid + ele] = dot_prod<T,4,4,1>(&s_Xworld[16*par + row], &s_Xhom[16*jid + 4*col]);
            }
        }
        __syncthreads();
        // BFS level 4 -> joints [4]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 16; ind += blockDim.x*blockDim.y){
            int slot = ind / 16; int ele = ind % 16;
            int row = ele & 3; int col = ele >> 2;
            // branch to get pointer locations
            int jid; int par;
                 if (slot < 1){ jid = 4; par = 3; }
            if (par == -1) {
                s_Xworld[16*jid + ele] = s_Xhom[16*jid + ele];
            }
            else {
                s_Xworld[16*jid + ele] = dot_prod<T,4,4,1>(&s_Xworld[16*par + row], &s_Xhom[16*jid + 4*col]);
            }
        }
        __syncthreads();
        // BFS level 5 -> joints [5]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 16; ind += blockDim.x*blockDim.y){
            int slot = ind / 16; int ele = ind % 16;
            int row = ele & 3; int col = ele >> 2;
            // branch to get pointer locations
            int jid; int par;
                 if (slot < 1){ jid = 5; par = 4; }
            if (par == -1) {
                s_Xworld[16*jid + ele] = s_Xhom[16*jid + ele];
            }
            else {
                s_Xworld[16*jid + ele] = dot_prod<T,4,4,1>(&s_Xworld[16*par + row], &s_Xhom[16*jid + 4*col]);
            }
        }
        __syncthreads();
        // BFS level 6 -> joints [6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 16; ind += blockDim.x*blockDim.y){
            int slot = ind / 16; int ele = ind % 16;
            int row = ele & 3; int col = ele >> 2;
            // branch to get pointer locations
            int jid; int par;
                 if (slot < 1){ jid = 6; par = 5; }
            if (par == -1) {
                s_Xworld[16*jid + ele] = s_Xhom[16*jid + ele];
            }
            else {
                s_Xworld[16*jid + ele] = dot_prod<T,4,4,1>(&s_Xworld[16*par + row], &s_Xhom[16*jid + 4*col]);
            }
        }
        __syncthreads();
        //
        // Step 1b: world transform of the fixed kinematic target(s): Xworld[anchor] = Xworld[parent] @ Xhom_local[anchor]
        //
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 16; ind += blockDim.x*blockDim.y){
            int slot = ind / 16; int ele = ind % 16;
            int row = ele & 3; int col = ele >> 2;
            // branch to get pointer locations
            int anc; int par;
                 if (slot < 1){ anc = 7; par = 6; }
            s_Xworld[16*anc + ele] = dot_prod<T,4,4,1>(&s_Xworld[16*par + row], &s_Xhom[16*anc + 4*col]);
        }
        __syncthreads();
        //
        // Step 2: zero the J_v and J_w scratch (out-of-chain columns stay zero)
        //
        glass::set_const<T, 42>(static_cast<T>(0), s_Jv);
        //
        // Step 3: per-chain-joint columns of J_v, J_w (one block-parallel work-item per (ee, S-column))
        //
        static const int eeg_job_j[] = { 0, 1, 2, 3, 4, 5, 6 };
        static const int eeg_job_anc[] = { 7, 7, 7, 7, 7, 7, 7 };
        static const int eeg_job_rev[] = { 1, 1, 1, 1, 1, 1, 1 };
        static const int eeg_job_base[] = { 0, 3, 6, 9, 12, 15, 18 };
        static const T eeg_job_ax[] = { static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(1) };
        for(int job_idx = threadIdx.x + threadIdx.y*blockDim.x; job_idx < 7; job_idx += blockDim.x*blockDim.y){
            int j   = eeg_job_j[job_idx];
            int ee_anchor = eeg_job_anc[job_idx];
            int col_base = eeg_job_base[job_idx];
            T ax0 = eeg_job_ax[3*job_idx + 0]; T ax1 = eeg_job_ax[3*job_idx + 1]; T ax2 = eeg_job_ax[3*job_idx + 2];
            T axw_0 = s_Xworld[16*j + 0]*ax0 + s_Xworld[16*j + 4]*ax1 + s_Xworld[16*j + 8]*ax2;
            T axw_1 = s_Xworld[16*j + 1]*ax0 + s_Xworld[16*j + 5]*ax1 + s_Xworld[16*j + 9]*ax2;
            T axw_2 = s_Xworld[16*j + 2]*ax0 + s_Xworld[16*j + 6]*ax1 + s_Xworld[16*j + 10]*ax2;
            if (eeg_job_rev[job_idx]) {
                s_Jw[col_base + 0] = axw_0; s_Jw[col_base + 1] = axw_1; s_Jw[col_base + 2] = axw_2;
                T dx = s_Xworld[16*ee_anchor + 12] - s_Xworld[16*j + 12];
                T dy = s_Xworld[16*ee_anchor + 13] - s_Xworld[16*j + 13];
                T dz = s_Xworld[16*ee_anchor + 14] - s_Xworld[16*j + 14];
                s_Jv[col_base + 0] = axw_1*dz - axw_2*dy;
                s_Jv[col_base + 1] = axw_2*dx - axw_0*dz;
                s_Jv[col_base + 2] = axw_0*dy - axw_1*dx;
            }
            else {
                s_Jv[col_base + 0] = axw_0; s_Jv[col_base + 1] = axw_1; s_Jv[col_base + 2] = axw_2;
            }
        }
        __syncthreads();
        //
        // Step 4: extract (cy, sy, cp, sp) from each ee's world rotation for E(rpy)^{-1}
        //
        for(int ee = threadIdx.x + threadIdx.y*blockDim.x; ee < 1; ee += blockDim.x*blockDim.y){
            const int ee_jid = 7;
            T R20 = s_Xworld[16*ee_jid + 2];
            T R21 = s_Xworld[16*ee_jid + 6];
            T R22 = s_Xworld[16*ee_jid + 10];
            T R10 = s_Xworld[16*ee_jid + 1];
            T R00 = s_Xworld[16*ee_jid + 0];
            T cp_term = sqrt(R22*R22 + R21*R21);
            T yaw = atan2(R10, R00);
            T pitch = atan2(-R20, cp_term);
            s_E_sc[4*ee + 0] = cos(yaw);
            s_E_sc[4*ee + 1] = sin(yaw);
            s_E_sc[4*ee + 2] = cos(pitch);
            s_E_sc[4*ee + 3] = sin(pitch);
        }
        __syncthreads();
        //
        // Step 5: write s_end_effector_pose_gradient (rows 0..2 = J_v, rows 3..5 = E(rpy)^{-1} J_w)
        //
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 42; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int rem = ind / 6; int vi = rem % 7; int ee = rem / 7;
            int jv_base = 3 * (7 * ee + vi);
            if (row < 3) {
                s_end_effector_pose_gradient[ind] = s_Jv[jv_base + row];
            }
            else {
                T cy = s_E_sc[4*ee + 0]; T sy = s_E_sc[4*ee + 1]; T cp = s_E_sc[4*ee + 2]; T sp = s_E_sc[4*ee + 3];
                T Jw0 = s_Jw[jv_base + 0]; T Jw1 = s_Jw[jv_base + 1]; T Jw2 = s_Jw[jv_base + 2];
                T outv;
                if (row == 3) { outv = (cy*Jw0 + sy*Jw1) / cp; }
                else if (row == 4) { outv = -sy*Jw0 + cy*Jw1; }
                else { outv = (sp / cp) * (cy*Jw0 + sy*Jw1) + Jw2; }
                s_end_effector_pose_gradient[ind] = outv;
            }
        }
        __syncthreads();
    }

    /**
     * Computes the Gradient of the End Effector Pose with respect to joint position
     *
     * @param s_end_effector_pose_gradient is a pointer to shared memory of size 6*NUM_VEL*NUM_EE where NUM_VEL = 7 and NUM_EE = 1
     * @param s_q is the vector of joint positions
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     */
    template <typename T>
    __device__
    void end_effector_pose_gradient_device_EE(T *s_end_effector_pose_gradient, const T *s_q, const robotModel<T> *d_robotModel) {
        // GRID shared arena layout
        //   T s_XmatsHom[144]
        //   T s_temp[190]
        //   bytes s_linalg_smem[GRID_EE_LINALG_SHARED_BYTES<T>()]
        extern __shared__ __align__(16) unsigned char s_arena[];
        size_t s_arena_offset = 0;
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_XmatsHom = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(144);
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_temp = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(190);
        int *s_topology_helpers = nullptr;
        unsigned char *s_linalg_smem = nullptr;
        if (static_cast<size_t>(GRID_EE_LINALG_SHARED_BYTES<T>()) > 0) {
            s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
            s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
            s_arena_offset += static_cast<size_t>(GRID_EE_LINALG_SHARED_BYTES<T>());
        }
        #ifdef GRID_CUDA_DEBUG_LAYOUT
        assert(s_arena_offset == grid_shared_arena_bytes<T>(334, 0, GRID_EE_LINALG_SHARED_BYTES<T>()));
        #endif
        (void)s_arena_offset;
        load_update_XmatsHom_helpers<T>(s_XmatsHom, s_topology_helpers, s_q, d_robotModel, s_temp);
        end_effector_pose_gradient_inner_EE<T, true>(s_end_effector_pose_gradient, s_q, s_XmatsHom, nullptr, s_topology_helpers, s_temp, nullptr, s_linalg_smem);
    }

    /**
     * Computes the Gradient of the End Effector Pose with respect to joint position
     *
     * @param d_end_effector_pose_gradient is the vector of end effector positions gradients
     * @param d_workspace is the generated global spill workspace
     * @param d_q is the vector of joint positions
     * @param stride_q is the stide between each q
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER, bool MUJOCO_OUTPUT = false>
    __global__
    __launch_bounds__(tier_max_threads<RESOURCE_TIER>())
    void end_effector_pose_gradient_kernel_EE_single_timing(T *d_end_effector_pose_gradient, unsigned char *d_workspace, const T *d_q, const int stride_q, const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS) {
        if constexpr (RESOURCE_TIER == TIER_SHARED) {
            // GRID shared arena layout
            //   T s_q[7]
            //   T s_end_effector_pose_gradient[42]
            //   T s_XmatsHom[144]
            //   T s_temp[190]
            //   bytes s_linalg_smem[GRID_EE_LINALG_SHARED_BYTES<T>()]
            extern __shared__ __align__(16) unsigned char s_arena[];
            size_t s_arena_offset = 0;
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_q = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(7);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_end_effector_pose_gradient = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(42);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_XmatsHom = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(144);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_temp = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(190);
            int *s_topology_helpers = nullptr;
            unsigned char *s_linalg_smem = nullptr;
            if (static_cast<size_t>(GRID_EE_LINALG_SHARED_BYTES<T>()) > 0) {
                s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
                s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
                s_arena_offset += static_cast<size_t>(GRID_EE_LINALG_SHARED_BYTES<T>());
            }
            #ifdef GRID_CUDA_DEBUG_LAYOUT
            assert(s_arena_offset == grid_shared_arena_bytes<T>(383, 0, GRID_EE_LINALG_SHARED_BYTES<T>()));
            #endif
            (void)s_arena_offset;
            (void)d_workspace;
            // load to shared mem
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                s_q[ind] = d_q[ind];
            }
            __syncthreads();
            // compute with NUM_TIMESTEPS as NUM_REPS for timing
            for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
                // anti-LICM: volatile reload of inputs each rep
                for(int _aopt_i = threadIdx.x + threadIdx.y*blockDim.x; _aopt_i < 7; _aopt_i += blockDim.x*blockDim.y){
                    reinterpret_cast<volatile T *>(s_q)[_aopt_i] = reinterpret_cast<const volatile T *>(d_q)[_aopt_i];
                }
                __syncthreads();
                // anti-LICM (1/2): stomp one input slot with `rep`
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
                    reinterpret_cast<volatile T *>(s_q)[rep % (7)] = static_cast<T>(rep);
                }
                // anti-LICM (2/2): feedback prev rep's d_end_effector_pose_gradient into s_q (true loop-carried dep)
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
                    T _aopt_fb1 = reinterpret_cast<const volatile T *>(d_end_effector_pose_gradient)[(rep + 0x3FF) & 0x3FF];
                    T _aopt_fb2 = reinterpret_cast<const volatile T *>(d_end_effector_pose_gradient)[(rep + 0x3FE) & 0x3FF];
                    T _aopt_fb3 = reinterpret_cast<const volatile T *>(d_end_effector_pose_gradient)[(rep + 0x3FD) & 0x3FF];
                    reinterpret_cast<volatile T *>(s_q)[(rep + 1) % (7)] += _aopt_fb1;
                    reinterpret_cast<volatile T *>(s_q)[(rep + 2) % (7)] += _aopt_fb2;
                    reinterpret_cast<volatile T *>(s_q)[(rep + 3) % (7)] += _aopt_fb3;
                }
                __syncthreads();
                load_update_XmatsHom_helpers<T>(s_XmatsHom, s_topology_helpers, s_q, d_robotModel, s_temp);
                end_effector_pose_gradient_inner_EE<T, true>(s_end_effector_pose_gradient, s_q, s_XmatsHom, nullptr, s_topology_helpers, s_temp, nullptr, s_linalg_smem);
                __syncthreads();
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) { reinterpret_cast<volatile T *>(d_end_effector_pose_gradient)[rep & 1023] = reinterpret_cast<const volatile T *>(s_end_effector_pose_gradient)[rep & 7]; }
            }
            // save down to global
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 42; ind += blockDim.x*blockDim.y){
                d_end_effector_pose_gradient[ind] = s_end_effector_pose_gradient[ind];
            }
            __syncthreads();
        }
        else if constexpr (RESOURCE_TIER == TIER_LITE) {
            // GRID shared arena layout
            //   T s_q[7]
            //   T s_end_effector_pose_gradient[42]
            //   T s_XmatsHom[144]
            //   T s_temp[190]
            //   bytes s_linalg_smem[GRID_EE_LINALG_SHARED_BYTES<T>()]
            extern __shared__ __align__(16) unsigned char s_arena[];
            size_t s_arena_offset = 0;
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_q = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(7);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_end_effector_pose_gradient = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(42);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_XmatsHom = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(144);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_temp = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(190);
            int *s_topology_helpers = nullptr;
            unsigned char *s_linalg_smem = nullptr;
            if (static_cast<size_t>(GRID_EE_LINALG_SHARED_BYTES<T>()) > 0) {
                s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
                s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
                s_arena_offset += static_cast<size_t>(GRID_EE_LINALG_SHARED_BYTES<T>());
            }
            #ifdef GRID_CUDA_DEBUG_LAYOUT
            assert(s_arena_offset == grid_shared_arena_bytes<T>(383, 0, GRID_EE_LINALG_SHARED_BYTES<T>()));
            #endif
            (void)s_arena_offset;
            (void)d_workspace;
            // load to shared mem
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                s_q[ind] = d_q[ind];
            }
            __syncthreads();
            // compute with NUM_TIMESTEPS as NUM_REPS for timing
            for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
                // anti-LICM: volatile reload of inputs each rep
                for(int _aopt_i = threadIdx.x + threadIdx.y*blockDim.x; _aopt_i < 7; _aopt_i += blockDim.x*blockDim.y){
                    reinterpret_cast<volatile T *>(s_q)[_aopt_i] = reinterpret_cast<const volatile T *>(d_q)[_aopt_i];
                }
                __syncthreads();
                // anti-LICM (1/2): stomp one input slot with `rep`
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
                    reinterpret_cast<volatile T *>(s_q)[rep % (7)] = static_cast<T>(rep);
                }
                // anti-LICM (2/2): feedback prev rep's d_end_effector_pose_gradient into s_q (true loop-carried dep)
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
                    T _aopt_fb1 = reinterpret_cast<const volatile T *>(d_end_effector_pose_gradient)[(rep + 0x3FF) & 0x3FF];
                    T _aopt_fb2 = reinterpret_cast<const volatile T *>(d_end_effector_pose_gradient)[(rep + 0x3FE) & 0x3FF];
                    T _aopt_fb3 = reinterpret_cast<const volatile T *>(d_end_effector_pose_gradient)[(rep + 0x3FD) & 0x3FF];
                    reinterpret_cast<volatile T *>(s_q)[(rep + 1) % (7)] += _aopt_fb1;
                    reinterpret_cast<volatile T *>(s_q)[(rep + 2) % (7)] += _aopt_fb2;
                    reinterpret_cast<volatile T *>(s_q)[(rep + 3) % (7)] += _aopt_fb3;
                }
                __syncthreads();
                load_update_XmatsHom_helpers<T>(s_XmatsHom, s_topology_helpers, s_q, d_robotModel, s_temp);
                end_effector_pose_gradient_inner_EE<T, true>(s_end_effector_pose_gradient, s_q, s_XmatsHom, nullptr, s_topology_helpers, s_temp, nullptr, s_linalg_smem);
                __syncthreads();
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) { reinterpret_cast<volatile T *>(d_end_effector_pose_gradient)[rep & 1023] = reinterpret_cast<const volatile T *>(s_end_effector_pose_gradient)[rep & 7]; }
            }
            // save down to global
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 42; ind += blockDim.x*blockDim.y){
                d_end_effector_pose_gradient[ind] = s_end_effector_pose_gradient[ind];
            }
            __syncthreads();
        }
        else if constexpr (RESOURCE_TIER == TIER_MINIMAL) {
            // GRID shared arena layout
            //   T s_q[7]
            //   T s_XmatsHom[144]
            //   bytes s_linalg_smem[GRID_EE_LINALG_SHARED_BYTES<T>()]
            extern __shared__ __align__(16) unsigned char s_arena[];
            size_t s_arena_offset = 0;
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_q = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(7);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_XmatsHom = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(144);
            T *s_temp = nullptr;
            int *s_topology_helpers = nullptr;
            unsigned char *s_linalg_smem = nullptr;
            if (static_cast<size_t>(GRID_EE_LINALG_SHARED_BYTES<T>()) > 0) {
                s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
                s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
                s_arena_offset += static_cast<size_t>(GRID_EE_LINALG_SHARED_BYTES<T>());
            }
            #ifdef GRID_CUDA_DEBUG_LAYOUT
            assert(s_arena_offset == grid_shared_arena_bytes<T>(151, 0, GRID_EE_LINALG_SHARED_BYTES<T>()));
            #endif
            (void)s_arena_offset;
            // load to shared mem
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                s_q[ind] = d_q[ind];
            }
            __syncthreads();
            T *s_end_effector_pose_gradient = d_end_effector_pose_gradient;
            T *s_eegrad_temp = reinterpret_cast<T *>(&d_workspace[GRID_END_EFFECTOR_POSE_GRADIENT_WORKSPACE_DXHOM_OFFSET_BYTES<T>()]);
            s_temp = s_eegrad_temp;
            // compute with NUM_TIMESTEPS as NUM_REPS for timing
            for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
                // anti-LICM: volatile reload of inputs each rep
                for(int _aopt_i = threadIdx.x + threadIdx.y*blockDim.x; _aopt_i < 7; _aopt_i += blockDim.x*blockDim.y){
                    reinterpret_cast<volatile T *>(s_q)[_aopt_i] = reinterpret_cast<const volatile T *>(d_q)[_aopt_i];
                }
                __syncthreads();
                // anti-LICM (1/2): stomp one input slot with `rep`
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
                    reinterpret_cast<volatile T *>(s_q)[rep % (7)] = static_cast<T>(rep);
                }
                // anti-LICM (2/2): feedback prev rep's d_end_effector_pose_gradient into s_q (true loop-carried dep)
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
                    T _aopt_fb1 = reinterpret_cast<const volatile T *>(d_end_effector_pose_gradient)[(rep + 0x3FF) & 0x3FF];
                    T _aopt_fb2 = reinterpret_cast<const volatile T *>(d_end_effector_pose_gradient)[(rep + 0x3FE) & 0x3FF];
                    T _aopt_fb3 = reinterpret_cast<const volatile T *>(d_end_effector_pose_gradient)[(rep + 0x3FD) & 0x3FF];
                    reinterpret_cast<volatile T *>(s_q)[(rep + 1) % (7)] += _aopt_fb1;
                    reinterpret_cast<volatile T *>(s_q)[(rep + 2) % (7)] += _aopt_fb2;
                    reinterpret_cast<volatile T *>(s_q)[(rep + 3) % (7)] += _aopt_fb3;
                }
                __syncthreads();
                load_update_XmatsHom_helpers<T>(s_XmatsHom, s_topology_helpers, s_q, d_robotModel, s_temp);
                end_effector_pose_gradient_inner_EE<T, false>(s_end_effector_pose_gradient, s_q, s_XmatsHom, nullptr, s_topology_helpers, s_temp, s_eegrad_temp, s_linalg_smem);
                __syncthreads();
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) { reinterpret_cast<volatile T *>(d_end_effector_pose_gradient)[rep & 1023] = reinterpret_cast<const volatile T *>(s_end_effector_pose_gradient)[rep & 7]; }
            }
        }
    }

    /**
     * Computes the Gradient of the End Effector Pose with respect to joint position
     *
     * @param d_end_effector_pose_gradient is the vector of end effector positions gradients
     * @param d_workspace is the generated global spill workspace
     * @param d_q is the vector of joint positions
     * @param stride_q is the stide between each q
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER, bool MUJOCO_OUTPUT = false>
    __global__
    __launch_bounds__(tier_max_threads<RESOURCE_TIER>())
    void end_effector_pose_gradient_kernel_EE(T *d_end_effector_pose_gradient, unsigned char *d_workspace, const T *d_q, const int stride_q, const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS) {
        if constexpr (RESOURCE_TIER == TIER_SHARED) {
            // GRID shared arena layout
            //   T s_q[7]
            //   T s_end_effector_pose_gradient[42]
            //   T s_XmatsHom[144]
            //   T s_temp[190]
            //   bytes s_linalg_smem[GRID_EE_LINALG_SHARED_BYTES<T>()]
            extern __shared__ __align__(16) unsigned char s_arena[];
            size_t s_arena_offset = 0;
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_q = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(7);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_end_effector_pose_gradient = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(42);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_XmatsHom = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(144);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_temp = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(190);
            int *s_topology_helpers = nullptr;
            unsigned char *s_linalg_smem = nullptr;
            if (static_cast<size_t>(GRID_EE_LINALG_SHARED_BYTES<T>()) > 0) {
                s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
                s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
                s_arena_offset += static_cast<size_t>(GRID_EE_LINALG_SHARED_BYTES<T>());
            }
            #ifdef GRID_CUDA_DEBUG_LAYOUT
            assert(s_arena_offset == grid_shared_arena_bytes<T>(383, 0, GRID_EE_LINALG_SHARED_BYTES<T>()));
            #endif
            (void)s_arena_offset;
            (void)d_workspace;
            for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
                // load to shared mem
                const T *d_q_k = &d_q[k*stride_q];
                for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                    s_q[ind] = d_q_k[ind];
                }
                __syncthreads();
                // compute
                load_update_XmatsHom_helpers<T>(s_XmatsHom, s_topology_helpers, s_q, d_robotModel, s_temp);
                end_effector_pose_gradient_inner_EE<T, true>(s_end_effector_pose_gradient, s_q, s_XmatsHom, nullptr, s_topology_helpers, s_temp, nullptr, s_linalg_smem);
                __syncthreads();
                // save down to global
                T *d_end_effector_pose_gradient_k = &d_end_effector_pose_gradient[k*42];
                for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 42; ind += blockDim.x*blockDim.y){
                    d_end_effector_pose_gradient_k[ind] = s_end_effector_pose_gradient[ind];
                }
                __syncthreads();
            }
        }
        else if constexpr (RESOURCE_TIER == TIER_LITE) {
            // GRID shared arena layout
            //   T s_q[7]
            //   T s_end_effector_pose_gradient[42]
            //   T s_XmatsHom[144]
            //   T s_temp[190]
            //   bytes s_linalg_smem[GRID_EE_LINALG_SHARED_BYTES<T>()]
            extern __shared__ __align__(16) unsigned char s_arena[];
            size_t s_arena_offset = 0;
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_q = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(7);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_end_effector_pose_gradient = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(42);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_XmatsHom = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(144);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_temp = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(190);
            int *s_topology_helpers = nullptr;
            unsigned char *s_linalg_smem = nullptr;
            if (static_cast<size_t>(GRID_EE_LINALG_SHARED_BYTES<T>()) > 0) {
                s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
                s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
                s_arena_offset += static_cast<size_t>(GRID_EE_LINALG_SHARED_BYTES<T>());
            }
            #ifdef GRID_CUDA_DEBUG_LAYOUT
            assert(s_arena_offset == grid_shared_arena_bytes<T>(383, 0, GRID_EE_LINALG_SHARED_BYTES<T>()));
            #endif
            (void)s_arena_offset;
            (void)d_workspace;
            for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
                // load to shared mem
                const T *d_q_k = &d_q[k*stride_q];
                for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                    s_q[ind] = d_q_k[ind];
                }
                __syncthreads();
                // compute
                load_update_XmatsHom_helpers<T>(s_XmatsHom, s_topology_helpers, s_q, d_robotModel, s_temp);
                end_effector_pose_gradient_inner_EE<T, true>(s_end_effector_pose_gradient, s_q, s_XmatsHom, nullptr, s_topology_helpers, s_temp, nullptr, s_linalg_smem);
                __syncthreads();
                // save down to global
                T *d_end_effector_pose_gradient_k = &d_end_effector_pose_gradient[k*42];
                for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 42; ind += blockDim.x*blockDim.y){
                    d_end_effector_pose_gradient_k[ind] = s_end_effector_pose_gradient[ind];
                }
                __syncthreads();
            }
        }
        else if constexpr (RESOURCE_TIER == TIER_MINIMAL) {
            // GRID shared arena layout
            //   T s_q[7]
            //   T s_XmatsHom[144]
            //   bytes s_linalg_smem[GRID_EE_LINALG_SHARED_BYTES<T>()]
            extern __shared__ __align__(16) unsigned char s_arena[];
            size_t s_arena_offset = 0;
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_q = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(7);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_XmatsHom = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(144);
            T *s_temp = nullptr;
            int *s_topology_helpers = nullptr;
            unsigned char *s_linalg_smem = nullptr;
            if (static_cast<size_t>(GRID_EE_LINALG_SHARED_BYTES<T>()) > 0) {
                s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
                s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
                s_arena_offset += static_cast<size_t>(GRID_EE_LINALG_SHARED_BYTES<T>());
            }
            #ifdef GRID_CUDA_DEBUG_LAYOUT
            assert(s_arena_offset == grid_shared_arena_bytes<T>(151, 0, GRID_EE_LINALG_SHARED_BYTES<T>()));
            #endif
            (void)s_arena_offset;
            for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
                // load to shared mem
                const T *d_q_k = &d_q[k*stride_q];
                for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                    s_q[ind] = d_q_k[ind];
                }
                __syncthreads();
                T *s_end_effector_pose_gradient = &d_end_effector_pose_gradient[k*42];
                T *s_eegrad_temp = reinterpret_cast<T *>(&d_workspace[grid_workspace_slot()*GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>() + GRID_END_EFFECTOR_POSE_GRADIENT_WORKSPACE_DXHOM_OFFSET_BYTES<T>()]);
                s_temp = s_eegrad_temp;
                // compute
                load_update_XmatsHom_helpers<T>(s_XmatsHom, s_topology_helpers, s_q, d_robotModel, s_temp);
                end_effector_pose_gradient_inner_EE<T, false>(s_end_effector_pose_gradient, s_q, s_XmatsHom, nullptr, s_topology_helpers, s_temp, s_eegrad_temp, s_linalg_smem);
                __syncthreads();
            }
        }
    }

    /**
     * Computes the Gradient of the End Effector Pose with respect to joint position
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_COMPRESSED_MEM = false, gridDataKind KIND = GRID_DATA_ALL, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>
    __host__
    void end_effector_pose_gradient_EE(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps,
                                const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_KINEMATICS, "end_effector_pose_gradient requires all-data or kinematics gridData");
        // start code with memory transfer
        int stride_q;
        if (USE_COMPRESSED_MEM) {stride_q = NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q,hd_data->h_q,stride_q*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        else {stride_q = 3*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        gpuErrchkKernel();
        // then call the kernel
        gpuErrchk(grid_check_dynamic_shared_memory_bytes("end_effector_pose_gradient", END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()));
        const int _grid_ws_n = (hd_data->workspace_timestep_slots > 0 && hd_data->workspace_timestep_slots < num_timesteps) ? hd_data->workspace_timestep_slots : num_timesteps;
        if (GRID_END_EFFECTOR_POSE_GRADIENT_USES_WORKSPACE_TEMP_ANY) {gpuErrchk(grid_begin_l2_persisting(0, hd_data->d_workspace, GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()*static_cast<size_t>(_grid_ws_n)));}
        dim3 _ws_grid = block_dimms;
        if ((int)(_ws_grid.x*_ws_grid.y*_ws_grid.z) > _grid_ws_n) { _ws_grid = dim3(_grid_ws_n,1,1); }
        dim3 _grid_thr_clamped_10 = grid_host_clamp_threads((const void*)&end_effector_pose_gradient_kernel_EE<T, RESOURCE_TIER>, thread_dimms);
        if (USE_COMPRESSED_MEM) {end_effector_pose_gradient_kernel_EE<T, RESOURCE_TIER><<<_ws_grid,_grid_thr_clamped_10,END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()>>>(hd_data->d_end_effector_pose_gradient,hd_data->d_workspace,hd_data->d_q,stride_q,d_robotModel,num_timesteps);}
        else                    {end_effector_pose_gradient_kernel_EE<T, RESOURCE_TIER><<<_ws_grid,_grid_thr_clamped_10,END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()>>>(hd_data->d_end_effector_pose_gradient,hd_data->d_workspace,hd_data->d_q_qd_u,stride_q,d_robotModel,num_timesteps);}
        gpuErrchkKernel();
        if (GRID_END_EFFECTOR_POSE_GRADIENT_USES_WORKSPACE_TEMP_ANY) {gpuErrchk(grid_end_l2_persisting(0));}
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_end_effector_pose_gradient,hd_data->d_end_effector_pose_gradient,6*NUM_EES*NUM_VEL*num_timesteps*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchkKernel();
    }

    /**
     * Computes the Gradient of the End Effector Pose with respect to joint position
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_COMPRESSED_MEM = false, gridDataKind KIND = GRID_DATA_ALL, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>
    __host__
    void end_effector_pose_gradient_EE_single_timing(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps,
                                              const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_KINEMATICS, "end_effector_pose_gradient requires all-data or kinematics gridData");
        // start code with memory transfer
        int stride_q;
        if (USE_COMPRESSED_MEM) {stride_q = NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q,hd_data->h_q,stride_q*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        else {stride_q = 3*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        gpuErrchkKernel();
        // then call the kernel
        gpuErrchk(grid_check_dynamic_shared_memory_bytes("end_effector_pose_gradient", END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()));
        if (GRID_END_EFFECTOR_POSE_GRADIENT_USES_WORKSPACE_TEMP_ANY) {gpuErrchk(grid_begin_l2_persisting(0, hd_data->d_workspace, GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()));}
        struct timespec start, end; clock_gettime(CLOCK_MONOTONIC,&start);
        dim3 _grid_thr_clamped_11 = grid_host_clamp_threads((const void*)&end_effector_pose_gradient_kernel_EE<T, RESOURCE_TIER>, thread_dimms);
        if (USE_COMPRESSED_MEM) {end_effector_pose_gradient_kernel_EE<T, RESOURCE_TIER><<<block_dimms,_grid_thr_clamped_11,END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()>>>(hd_data->d_end_effector_pose_gradient,hd_data->d_workspace,hd_data->d_q,stride_q,d_robotModel,num_timesteps);}
        else                    {end_effector_pose_gradient_kernel_EE<T, RESOURCE_TIER><<<block_dimms,_grid_thr_clamped_11,END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()>>>(hd_data->d_end_effector_pose_gradient,hd_data->d_workspace,hd_data->d_q_qd_u,stride_q,d_robotModel,num_timesteps);}
        gpuErrchkKernel();
        clock_gettime(CLOCK_MONOTONIC,&end);
        if (GRID_END_EFFECTOR_POSE_GRADIENT_USES_WORKSPACE_TEMP_ANY) {gpuErrchk(grid_end_l2_persisting(0));}
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_end_effector_pose_gradient,hd_data->d_end_effector_pose_gradient,6*NUM_EES*NUM_VEL*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchkKernel();
        printf("Single Call END_EFFECTOR_POSE_GRADIENT %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(num_timesteps));
    }

    /**
     * Computes the Gradient of the End Effector Pose with respect to joint position
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_COMPRESSED_MEM = false, gridDataKind KIND = GRID_DATA_ALL, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>
    __host__
    void end_effector_pose_gradient_EE_compute_only(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps,
                                             const dim3 block_dimms, const dim3 thread_dimms) {
        static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_KINEMATICS, "end_effector_pose_gradient requires all-data or kinematics gridData");
        int stride_q = USE_COMPRESSED_MEM ? NUM_JOINTS: 3*NUM_JOINTS;
        // then call the kernel
        gpuErrchk(grid_check_dynamic_shared_memory_bytes("end_effector_pose_gradient", END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()));
        const int _grid_ws_n = (hd_data->workspace_timestep_slots > 0 && hd_data->workspace_timestep_slots < num_timesteps) ? hd_data->workspace_timestep_slots : num_timesteps;
        if (GRID_END_EFFECTOR_POSE_GRADIENT_USES_WORKSPACE_TEMP_ANY) {gpuErrchk(grid_begin_l2_persisting(0, hd_data->d_workspace, GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()*static_cast<size_t>(_grid_ws_n)));}
        dim3 _ws_grid = block_dimms;
        if ((int)(_ws_grid.x*_ws_grid.y*_ws_grid.z) > _grid_ws_n) { _ws_grid = dim3(_grid_ws_n,1,1); }
        dim3 _grid_thr_clamped_12 = grid_host_clamp_threads((const void*)&end_effector_pose_gradient_kernel_EE<T, RESOURCE_TIER>, thread_dimms);
        if (USE_COMPRESSED_MEM) {end_effector_pose_gradient_kernel_EE<T, RESOURCE_TIER><<<_ws_grid,_grid_thr_clamped_12,END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()>>>(hd_data->d_end_effector_pose_gradient,hd_data->d_workspace,hd_data->d_q,stride_q,d_robotModel,num_timesteps);}
        else                    {end_effector_pose_gradient_kernel_EE<T, RESOURCE_TIER><<<_ws_grid,_grid_thr_clamped_12,END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()>>>(hd_data->d_end_effector_pose_gradient,hd_data->d_workspace,hd_data->d_q_qd_u,stride_q,d_robotModel,num_timesteps);}
        gpuErrchkKernel();
        if (GRID_END_EFFECTOR_POSE_GRADIENT_USES_WORKSPACE_TEMP_ANY) {gpuErrchk(grid_end_l2_persisting(0));}
    }

    /**
     * Thread-per-sample forward kinematics: serial chain walk that fills the full cumulative (world-frame) joint transforms s_jointXforms from s_q. Robot-general (any DoF / joint types). Reads s_jointXforms[16*target_idx] for the world pose of frame target_idx.
     *
     * Notes:
     *   Assumes the constant (q-independent) cells of s_XmatsHom are pre-loaded; only the q-dependent cells are refreshed here.
     *
     * @param s_jointXforms is the pointer to the cumulative (world) joint transforms (16 per joint)
     * @param s_XmatsHom is the pointer to the per-joint homogeneous transforms (16 per joint)
     * @param s_q is the vector of joint positions
     * @param target_idx is the joint index whose world transform is desired (full array is filled)
     */
    template <typename T>
    __device__
    void ee_pose_inner_thread(T *s_jointXforms, T *s_XmatsHom, T *s_q, int target_idx) {
        (void)target_idx;
        // X_hom[0] q-dependent cells
        {
            const T s = static_cast<T>(sin(s_q[0]));
            const T c = static_cast<T>(cos(s_q[0]));
            (void)s; (void)c;
            s_XmatsHom[16*0 + 0] = static_cast<T>(c);
            s_XmatsHom[16*0 + 1] = static_cast<T>(s);
            s_XmatsHom[16*0 + 4] = static_cast<T>(-s);
            s_XmatsHom[16*0 + 5] = static_cast<T>(c);
        }
        // X_hom[1] q-dependent cells
        {
            const T s = static_cast<T>(sin(s_q[1]));
            const T c = static_cast<T>(cos(s_q[1]));
            (void)s; (void)c;
            s_XmatsHom[16*1 + 0] = static_cast<T>(s);
            s_XmatsHom[16*1 + 1] = static_cast<T>(1.6098233857064764e-15*s);
            s_XmatsHom[16*1 + 2] = static_cast<T>(c);
            s_XmatsHom[16*1 + 4] = static_cast<T>(c);
            s_XmatsHom[16*1 + 5] = static_cast<T>(1.6098233857064764e-15*c);
            s_XmatsHom[16*1 + 6] = static_cast<T>(-s);
        }
        // X_hom[2] q-dependent cells
        {
            const T s = static_cast<T>(sin(s_q[2]));
            const T c = static_cast<T>(cos(s_q[2]));
            (void)s; (void)c;
            s_XmatsHom[16*2 + 1] = static_cast<T>(c);
            s_XmatsHom[16*2 + 2] = static_cast<T>(s);
            s_XmatsHom[16*2 + 5] = static_cast<T>(-s);
            s_XmatsHom[16*2 + 6] = static_cast<T>(c);
        }
        // X_hom[3] q-dependent cells
        {
            const T s = static_cast<T>(sin(s_q[3]));
            const T c = static_cast<T>(cos(s_q[3]));
            (void)s; (void)c;
            s_XmatsHom[16*3 + 0] = static_cast<T>(c);
            s_XmatsHom[16*3 + 2] = static_cast<T>(s);
            s_XmatsHom[16*3 + 4] = static_cast<T>(-s);
            s_XmatsHom[16*3 + 6] = static_cast<T>(c);
        }
        // X_hom[4] q-dependent cells
        {
            const T s = static_cast<T>(sin(s_q[4]));
            const T c = static_cast<T>(cos(s_q[4]));
            (void)s; (void)c;
            s_XmatsHom[16*4 + 0] = static_cast<T>(c);
            s_XmatsHom[16*4 + 2] = static_cast<T>(-s);
            s_XmatsHom[16*4 + 4] = static_cast<T>(-s);
            s_XmatsHom[16*4 + 6] = static_cast<T>(-c);
        }
        // X_hom[5] q-dependent cells
        {
            const T s = static_cast<T>(sin(s_q[5]));
            const T c = static_cast<T>(cos(s_q[5]));
            (void)s; (void)c;
            s_XmatsHom[16*5 + 0] = static_cast<T>(s);
            s_XmatsHom[16*5 + 1] = static_cast<T>(1.6098233857064764e-15*s);
            s_XmatsHom[16*5 + 2] = static_cast<T>(c);
            s_XmatsHom[16*5 + 4] = static_cast<T>(c);
            s_XmatsHom[16*5 + 5] = static_cast<T>(1.6098233857064764e-15*c);
            s_XmatsHom[16*5 + 6] = static_cast<T>(-s);
        }
        // X_hom[6] q-dependent cells
        {
            const T s = static_cast<T>(sin(s_q[6]));
            const T c = static_cast<T>(cos(s_q[6]));
            (void)s; (void)c;
            s_XmatsHom[16*6 + 1] = static_cast<T>(c);
            s_XmatsHom[16*6 + 2] = static_cast<T>(s);
            s_XmatsHom[16*6 + 5] = static_cast<T>(-s);
            s_XmatsHom[16*6 + 6] = static_cast<T>(c);
        }
        // chain up cumulative world transforms (parent always < child)
        #pragma unroll
        for (int j = 0; j < 7; ++j) {
            const T* c = &s_XmatsHom[j * 16];
            T* o = &s_jointXforms[j * 16];
            int par = j - 1;
            if (par < 0) {
                o[0]=c[0];   o[1]=c[1];   o[2]=c[2];
                o[4]=c[4];   o[5]=c[5];   o[6]=c[6];
                o[8]=c[8];   o[9]=c[9];   o[10]=c[10];
                o[12]=c[12]; o[13]=c[13]; o[14]=c[14];
            }
            else {
                const T* p = &s_jointXforms[par * 16];
                o[0]  = p[0]*c[0]   + p[4]*c[1]   + p[8]*c[2];
                o[1]  = p[1]*c[0]   + p[5]*c[1]   + p[9]*c[2];
                o[2]  = p[2]*c[0]   + p[6]*c[1]   + p[10]*c[2];
                o[4]  = p[0]*c[4]   + p[4]*c[5]   + p[8]*c[6];
                o[5]  = p[1]*c[4]   + p[5]*c[5]   + p[9]*c[6];
                o[6]  = p[2]*c[4]   + p[6]*c[5]   + p[10]*c[6];
                o[8]  = p[0]*c[8]   + p[4]*c[9]   + p[8]*c[10];
                o[9]  = p[1]*c[8]   + p[5]*c[9]   + p[9]*c[10];
                o[10] = p[2]*c[8]   + p[6]*c[9]   + p[10]*c[10];
                o[12] = p[0]*c[12]  + p[4]*c[13]  + p[8]*c[14]  + p[12];
                o[13] = p[1]*c[12]  + p[5]*c[13]  + p[9]*c[14]  + p[13];
                o[14] = p[2]*c[12]  + p[6]*c[13]  + p[10]*c[14] + p[14];
            }
            o[3]=(T)0; o[7]=(T)0; o[11]=(T)0; o[15]=(T)1;
        }
    }

    /**
     * Single-joint homogeneous-transform builder: writes joint j's 4x4 local transform (16 cells) into s_Xj at an EXPLICIT angle theta, sourcing the constant cells from the pre-loaded s_XmatsHom and overriding only the q-dependent cells. Robot-general (switch over the parser's per-joint Xmats). Lets a caller re-evaluate one perturbed joint without recomputing the whole chain (coordinate-descent candidate FK / suffix recompute).
     *
     * Notes:
     *   Assumes the constant (q-independent) cells of s_XmatsHom are pre-loaded.
     *
     * @param s_Xj is the 16-element destination for joint j's local transform
     * @param s_XmatsHom is the pointer to the per-joint homogeneous transforms (16 per joint)
     * @param j is the joint id to (re)build
     * @param theta is the joint angle to evaluate at
     */
    template <typename T>
    __device__ __forceinline__
    void update_XmatHom_joint(T *s_Xj, const T *s_XmatsHom, int j, T theta) {
        const T s = static_cast<T>(sin(theta));
        const T c = static_cast<T>(cos(theta));
        (void)s; (void)c;
        #pragma unroll
        for (int m = 0; m < 16; ++m) { s_Xj[m] = s_XmatsHom[16*j + m]; }
        switch (j) {
            case 0: {
                s_Xj[0] = static_cast<T>(c);
                s_Xj[1] = static_cast<T>(s);
                s_Xj[4] = static_cast<T>(-s);
                s_Xj[5] = static_cast<T>(c);
            } break;
            case 1: {
                s_Xj[0] = static_cast<T>(s);
                s_Xj[1] = static_cast<T>(1.6098233857064764e-15*s);
                s_Xj[2] = static_cast<T>(c);
                s_Xj[4] = static_cast<T>(c);
                s_Xj[5] = static_cast<T>(1.6098233857064764e-15*c);
                s_Xj[6] = static_cast<T>(-s);
            } break;
            case 2: {
                s_Xj[1] = static_cast<T>(c);
                s_Xj[2] = static_cast<T>(s);
                s_Xj[5] = static_cast<T>(-s);
                s_Xj[6] = static_cast<T>(c);
            } break;
            case 3: {
                s_Xj[0] = static_cast<T>(c);
                s_Xj[2] = static_cast<T>(s);
                s_Xj[4] = static_cast<T>(-s);
                s_Xj[6] = static_cast<T>(c);
            } break;
            case 4: {
                s_Xj[0] = static_cast<T>(c);
                s_Xj[2] = static_cast<T>(-s);
                s_Xj[4] = static_cast<T>(-s);
                s_Xj[6] = static_cast<T>(-c);
            } break;
            case 5: {
                s_Xj[0] = static_cast<T>(s);
                s_Xj[1] = static_cast<T>(1.6098233857064764e-15*s);
                s_Xj[2] = static_cast<T>(c);
                s_Xj[4] = static_cast<T>(c);
                s_Xj[5] = static_cast<T>(1.6098233857064764e-15*c);
                s_Xj[6] = static_cast<T>(-s);
            } break;
            case 6: {
                s_Xj[1] = static_cast<T>(c);
                s_Xj[2] = static_cast<T>(s);
                s_Xj[5] = static_cast<T>(-s);
                s_Xj[6] = static_cast<T>(c);
            } break;
            default: break;
        }
    }

    /**
     * Warp-per-sample forward kinematics: warp-cooperative chain walk that fills the full cumulative (world-frame) joint transforms s_jointXforms from s_q. Robot-general (any DoF / joint types). 3 lanes own the matrix rows; __syncwarp between levels. Reads s_jointXforms[16*target_idx] for the world pose of frame target_idx.
     *
     * Notes:
     *   Assumes the constant (q-independent) cells of s_XmatsHom are pre-loaded; only the q-dependent cells are refreshed here.
     *
     * @param s_jointXforms is the pointer to the cumulative (world) joint transforms (16 per joint)
     * @param s_XmatsHom is the pointer to the per-joint homogeneous transforms (16 per joint)
     * @param s_q is the vector of joint positions
     * @param target_idx is the joint index whose world transform is desired (full array is filled)
     */
    template <typename T>
    __device__ inline void ee_pose_inner_warp(
        T* __restrict__ s_jointXforms,
        T* __restrict__ s_XmatsHom,
        const T* __restrict__ s_q,
        int target_idx)
    {
        (void)target_idx;
        const int lane = threadIdx.x & 31;
        const unsigned mask = 0xFFFFFFFFu;
        // refresh q-dependent X_hom cells: lane j owns joint j
        if (lane == 0) {
            const T s = static_cast<T>(sin(s_q[0]));
            const T c = static_cast<T>(cos(s_q[0]));
            (void)s; (void)c;
            s_XmatsHom[16*0 + 0] = static_cast<T>(c);
            s_XmatsHom[16*0 + 1] = static_cast<T>(s);
            s_XmatsHom[16*0 + 4] = static_cast<T>(-s);
            s_XmatsHom[16*0 + 5] = static_cast<T>(c);
        }
        if (lane == 1) {
            const T s = static_cast<T>(sin(s_q[1]));
            const T c = static_cast<T>(cos(s_q[1]));
            (void)s; (void)c;
            s_XmatsHom[16*1 + 0] = static_cast<T>(s);
            s_XmatsHom[16*1 + 1] = static_cast<T>(1.6098233857064764e-15*s);
            s_XmatsHom[16*1 + 2] = static_cast<T>(c);
            s_XmatsHom[16*1 + 4] = static_cast<T>(c);
            s_XmatsHom[16*1 + 5] = static_cast<T>(1.6098233857064764e-15*c);
            s_XmatsHom[16*1 + 6] = static_cast<T>(-s);
        }
        if (lane == 2) {
            const T s = static_cast<T>(sin(s_q[2]));
            const T c = static_cast<T>(cos(s_q[2]));
            (void)s; (void)c;
            s_XmatsHom[16*2 + 1] = static_cast<T>(c);
            s_XmatsHom[16*2 + 2] = static_cast<T>(s);
            s_XmatsHom[16*2 + 5] = static_cast<T>(-s);
            s_XmatsHom[16*2 + 6] = static_cast<T>(c);
        }
        if (lane == 3) {
            const T s = static_cast<T>(sin(s_q[3]));
            const T c = static_cast<T>(cos(s_q[3]));
            (void)s; (void)c;
            s_XmatsHom[16*3 + 0] = static_cast<T>(c);
            s_XmatsHom[16*3 + 2] = static_cast<T>(s);
            s_XmatsHom[16*3 + 4] = static_cast<T>(-s);
            s_XmatsHom[16*3 + 6] = static_cast<T>(c);
        }
        if (lane == 4) {
            const T s = static_cast<T>(sin(s_q[4]));
            const T c = static_cast<T>(cos(s_q[4]));
            (void)s; (void)c;
            s_XmatsHom[16*4 + 0] = static_cast<T>(c);
            s_XmatsHom[16*4 + 2] = static_cast<T>(-s);
            s_XmatsHom[16*4 + 4] = static_cast<T>(-s);
            s_XmatsHom[16*4 + 6] = static_cast<T>(-c);
        }
        if (lane == 5) {
            const T s = static_cast<T>(sin(s_q[5]));
            const T c = static_cast<T>(cos(s_q[5]));
            (void)s; (void)c;
            s_XmatsHom[16*5 + 0] = static_cast<T>(s);
            s_XmatsHom[16*5 + 1] = static_cast<T>(1.6098233857064764e-15*s);
            s_XmatsHom[16*5 + 2] = static_cast<T>(c);
            s_XmatsHom[16*5 + 4] = static_cast<T>(c);
            s_XmatsHom[16*5 + 5] = static_cast<T>(1.6098233857064764e-15*c);
            s_XmatsHom[16*5 + 6] = static_cast<T>(-s);
        }
        if (lane == 6) {
            const T s = static_cast<T>(sin(s_q[6]));
            const T c = static_cast<T>(cos(s_q[6]));
            (void)s; (void)c;
            s_XmatsHom[16*6 + 1] = static_cast<T>(c);
            s_XmatsHom[16*6 + 2] = static_cast<T>(s);
            s_XmatsHom[16*6 + 5] = static_cast<T>(-s);
            s_XmatsHom[16*6 + 6] = static_cast<T>(c);
        }
        __syncwarp(mask);
        // chain up cumulative world transforms (parent always < child)
        #pragma unroll
        for (int j = 0; j < 7; ++j) {
            const T* c = &s_XmatsHom[j * 16];
            T* o = &s_jointXforms[j * 16];
            int par = j - 1;
            if (par < 0) {
                if (lane == 0) { o[0]=c[0]; o[4]=c[4]; o[8]=c[8];  o[12]=c[12]; o[3]=(T)0; o[7]=(T)0; o[11]=(T)0; o[15]=(T)1; }
                else if (lane == 1) { o[1]=c[1]; o[5]=c[5]; o[9]=c[9];  o[13]=c[13]; }
                else if (lane == 2) { o[2]=c[2]; o[6]=c[6]; o[10]=c[10]; o[14]=c[14]; }
            }
            else {
                const T* p = &s_jointXforms[par * 16];
                if (lane == 0) {
                    o[0]  = p[0]*c[0]  + p[4]*c[1]  + p[8]*c[2];
                    o[4]  = p[0]*c[4]  + p[4]*c[5]  + p[8]*c[6];
                    o[8]  = p[0]*c[8]  + p[4]*c[9]  + p[8]*c[10];
                    o[12] = p[0]*c[12] + p[4]*c[13] + p[8]*c[14] + p[12];
                    o[3]  = (T)0;      o[7]  = (T)0;      o[11] = (T)0;      o[15] = (T)1;
                }
                else if (lane == 1) {
                    o[1]  = p[1]*c[0]  + p[5]*c[1]  + p[9]*c[2];
                    o[5]  = p[1]*c[4]  + p[5]*c[5]  + p[9]*c[6];
                    o[9]  = p[1]*c[8]  + p[5]*c[9]  + p[9]*c[10];
                    o[13] = p[1]*c[12] + p[5]*c[13] + p[9]*c[14] + p[13];
                }
                else if (lane == 2) {
                    o[2]  = p[2]*c[0]  + p[6]*c[1]  + p[10]*c[2];
                    o[6]  = p[2]*c[4]  + p[6]*c[5]  + p[10]*c[6];
                    o[10] = p[2]*c[8]  + p[6]*c[9]  + p[10]*c[10];
                    o[14] = p[2]*c[12] + p[6]*c[13] + p[10]*c[14] + p[14];
                }
            }
            __syncwarp(mask);
        }
    }

    /**
     * Batched forward kinematics: one block per sample, pos+quat output.
     *
     * Notes:
     *   USE_WARP picks the warp-cooperative inner (warp 0) vs the thread inner (thread 0).
     *   target_idx selects the output frame (defaults to the leaf EE joint id).
     *
     * @param d_pose7 is the (B x 7) output: [tx,ty,tz, qw,qx,qy,qz] per sample
     * @param d_q is the (B x NUM_POS) joint-position input (batch-major, stride stride_q)
     * @param stride_q is the stride between samples in d_q (== NUM_POS)
     * @param d_robotModel holds the per-robot constants (XImats, topology helpers)
     * @param B is the batch size (== gridDim.x)
     * @param target_idx is the joint frame whose world pose is written
     */
    template <typename T, bool USE_WARP = false>
    __global__
    void ee_pose_fk_batched_kernel(T *d_pose7, const T *d_q, const int stride_q, const robotModel<T> *d_robotModel, const int B, const int target_idx = 6) {
        __shared__ T s_q[7];
        __shared__ T s_XmatsHom[144];
        __shared__ T s_jointXforms[112];
        __shared__ T s_temp[14];
        int *s_topology_helpers = nullptr;
        for (int b = blockIdx.x; b < B; b += gridDim.x) {
            for (int j = threadIdx.x; j < 7; j += blockDim.x) { s_q[j] = d_q[b*stride_q + j]; }
            __syncthreads();
            load_update_XmatsHom_helpers<T>(s_XmatsHom, s_topology_helpers, s_q, d_robotModel, s_temp);
            __syncthreads();
            if (USE_WARP) {
                if ((threadIdx.x >> 5) == 0) { ee_pose_inner_warp<T>(s_jointXforms, s_XmatsHom, s_q, target_idx); }
            }
            else {
                if (threadIdx.x == 0) { ee_pose_inner_thread<T>(s_jointXforms, s_XmatsHom, s_q, target_idx); }
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                const T* X = &s_jointXforms[16*target_idx];
                // rotation block (column-major): R[r][c] = X[4*c + r]
                const T r00=X[0], r10=X[1], r20=X[2];
                const T r01=X[4], r11=X[5], r21=X[6];
                const T r02=X[8], r12=X[9], r22=X[10];
                T* o = &d_pose7[b*7];
                o[0]=X[12]; o[1]=X[13]; o[2]=X[14];
                // quaternion (w,x,y,z) from rotation (Shepperd's method)
                const T tr = r00 + r11 + r22;
                T qw,qx,qy,qz;
                if (tr > (T)0) {
                    T S = sqrt(tr + (T)1) * (T)2; qw = (T)0.25*S; qx=(r21-r12)/S; qy=(r02-r20)/S; qz=(r10-r01)/S;
                }
                else if (r00 > r11 && r00 > r22) {
                    T S = sqrt((T)1 + r00 - r11 - r22) * (T)2; qw=(r21-r12)/S; qx=(T)0.25*S; qy=(r01+r10)/S; qz=(r02+r20)/S;
                }
                else if (r11 > r22) {
                    T S = sqrt((T)1 + r11 - r00 - r22) * (T)2; qw=(r02-r20)/S; qx=(r01+r10)/S; qy=(T)0.25*S; qz=(r12+r21)/S;
                }
                else {
                    T S = sqrt((T)1 + r22 - r00 - r11) * (T)2; qw=(r10-r01)/S; qx=(r02+r20)/S; qy=(r12+r21)/S; qz=(T)0.25*S;
                }
                o[3]=qw; o[4]=qx; o[5]=qy; o[6]=qz;
            }
            __syncthreads();
        }
    }

    /**
     * Host launcher for batched FK (<<<B, threads>>>, one block per sample).
     *
     * Notes:
     *   USE_WARP selects the warp- vs thread-cooperative per-sample inner.
     *
     * @param d_pose7 is the device (B x 7) output buffer
     * @param d_q is the device (B x NUM_POS) input buffer (batch-major)
     * @param B is the batch size
     * @param d_robotModel holds the per-robot constants
     * @param threads is the per-block thread count (>=32 for the warp variant)
     * @param target_idx selects the output frame (defaults to the leaf EE)
     * @param stream is the CUDA stream
     */
    template <typename T, bool USE_WARP = false>
    __host__
    void ee_pose_fk_batched(T *d_pose7, const T *d_q, const int B, const robotModel<T> *d_robotModel, const int threads = 32, const int target_idx = 6, cudaStream_t stream = (cudaStream_t)0) {
        const int stride_q = 7;
        ee_pose_fk_batched_kernel<T, USE_WARP><<<B, threads, 0, stream>>>(d_pose7, d_q, stride_q, d_robotModel, B, target_idx);
    }

    #define GRID_HAS_FK_BATCHED 1
    // ---- GATO Ask-4: stable end-effector TARGET aliases -------------------------
    // Resolve to the named fixed kinematic target when one was generated, else to the
    // generic (last-moving-joint) family. ALWAYS defined, so a consumer never has to
    // name a robot-specific joint (indy7 '_EE' vs panda '_panda_hand') or feature-test.
    // NAMED target_EE -> TRUE ee_frame (== pinocchio oMf[target]); NUM_EE = 1.
    // 
    const int NUM_TARGET_EES = 1;
    #define GRID_EE_FIXED_TARGET_NAME "EE"
    
    template <typename T, bool TEMP_IN_SMEM = true, typename... Args>
    __device__ __forceinline__
    void end_effector_pose_target_inner(Args... args) { end_effector_pose_inner_EE<T, TEMP_IN_SMEM>(args...); }
    
    template <typename T, typename... Args>
    __device__ __forceinline__
    void end_effector_pose_target_device(Args... args) { end_effector_pose_device_EE<T>(args...); }
    
    template <typename T, bool TEMP_IN_SMEM = true, typename... Args>
    __device__ __forceinline__
    void end_effector_pose_gradient_target_inner(Args... args) { end_effector_pose_gradient_inner_EE<T, TEMP_IN_SMEM>(args...); }
    
    template <typename T, typename... Args>
    __device__ __forceinline__
    void end_effector_pose_gradient_target_device(Args... args) { end_effector_pose_gradient_device_EE<T>(args...); }
    
    // ---- grid_rbd EE binding entry-point aliases (single named target routes here) ----
    #define GRID_RBD_NUM_EES 1
    #define GRID_RBD_EE_POSE_FN end_effector_pose_EE
    #define GRID_RBD_EE_POSE_GRADIENT_FN end_effector_pose_gradient_EE
    #define GRID_RBD_EE_POSE_HESSIAN_FN end_effector_pose_hessian_EE
    #define GRID_RBD_EE_POSE_KERNEL end_effector_pose_kernel_EE
    #define GRID_RBD_EE_POSE_GRADIENT_KERNEL end_effector_pose_gradient_kernel_EE
    #define GRID_RBD_EE_POSE_HESSIAN_KERNEL end_effector_pose_hessian_kernel_EE
    
    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * Notes:
     *   Assumes the XI matricies have already been updated for the given q
     *
     * @param s_c is the vector of output torques
     * @param s_vaf is a pointer to shared memory of size 3*6*NUM_JOINTS = 126
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param s_XI is the pointer to the transformation and inertia matricies 
     * @param s_qdd is (optional vector of joint accelerations
     * @param s_temp is a pointer to helper shared memory of size 6*NUM_JOINTS = 42
     * @param s_XImats is the (shared) memory holding the updated XI matricies for the given s_q
     * @param s_topology_helpers is the (shared) memory location for the topology_helpers (nullptr/unused for serial chains with identical Ss)
     * @param gravity is the gravity constant
     * @param d_f_ext is the (optional) GLOBAL external forces, body-major 6*NUM_BODIES local-frame, or nullptr
     */
    template <typename T>
    __device__
    void inverse_dynamics_inner(T *s_c,  T *s_vaf, const T *s_q, const T *s_qd, const T *s_qdd, T *s_XImats, int *s_topology_helpers, T *s_temp, T *d_f_ext, const T gravity) {
        unsigned char *s_linalg_smem = nullptr;
        //
        // Forward Pass
        //
        // s_v, s_a where parent is base
        //     joints are: A1
        //     links are: L1
        // s_v[k] = S[k]*qd[k] and s_a[k] = X[k]*gravityS[k]*qdd[k]
        for(int row = threadIdx.x + threadIdx.y*blockDim.x; row < 6; row += blockDim.x*blockDim.y){
            int jid6 = 6*0;
            s_vaf[jid6 + row] = static_cast<T>(0);
            s_vaf[42 + jid6 + row] = -s_XImats[6*jid6 + 30 + row]*gravity;
            if (row == 2){s_vaf[jid6 + 2] += (1) * s_qd[0]; s_vaf[42 + jid6 + 2] += (1) * s_qdd[0];}
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 1
        //     joints are: A2
        //     links are: L2
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + S[k]*qdd[k] + mxS[k](v[k])*qd[k]
        static const int seg_a_off_lvl1[1] = {36};
        static const int seg_v_x_off_lvl1[1] = {0};
        static const int seg_v_y_off_lvl1[1] = {6};
        static const int seg_s_off_lvl1[1] = {0};
        static const T S_sel_lvl1[6] = {static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0)};
        if(threadIdx.x == 0 && threadIdx.y == 0){
            s_temp[0] = s_qd[1];
        }
        __syncthreads();
        grid_linalg_segmented_row_strided_gemv<T,6,6,6,true>(1, seg_a_off_lvl1, seg_v_x_off_lvl1, seg_v_y_off_lvl1, s_XImats, s_vaf, s_vaf, static_cast<T>(1), static_cast<T>(0), seg_s_off_lvl1, S_sel_lvl1, s_temp, s_linalg_smem);
        static const int seg_a_x_off_lvl1[1] = {42};
        static const int seg_a_y_off_lvl1[1] = {48};
        if(threadIdx.x == 0 && threadIdx.y == 0){
            s_temp[0] = s_qdd[1];
        }
        __syncthreads();
        grid_linalg_segmented_row_strided_gemv<T,6,6,6,true>(1, seg_a_off_lvl1, seg_a_x_off_lvl1, seg_a_y_off_lvl1, s_XImats, s_vaf, s_vaf, static_cast<T>(1), static_cast<T>(0), seg_s_off_lvl1, S_sel_lvl1, s_temp, s_linalg_smem);
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[48], &s_vaf[6], (1) * s_qd[1]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 2
        //     joints are: A3
        //     links are: L3
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + S[k]*qdd[k] + mxS[k](v[k])*qd[k]
        static const int seg_a_off_lvl2[1] = {72};
        static const int seg_v_x_off_lvl2[1] = {6};
        static const int seg_v_y_off_lvl2[1] = {12};
        static const int seg_s_off_lvl2[1] = {0};
        static const T S_sel_lvl2[6] = {static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0)};
        if(threadIdx.x == 0 && threadIdx.y == 0){
            s_temp[0] = s_qd[2];
        }
        __syncthreads();
        grid_linalg_segmented_row_strided_gemv<T,6,6,6,true>(1, seg_a_off_lvl2, seg_v_x_off_lvl2, seg_v_y_off_lvl2, s_XImats, s_vaf, s_vaf, static_cast<T>(1), static_cast<T>(0), seg_s_off_lvl2, S_sel_lvl2, s_temp, s_linalg_smem);
        static const int seg_a_x_off_lvl2[1] = {48};
        static const int seg_a_y_off_lvl2[1] = {54};
        if(threadIdx.x == 0 && threadIdx.y == 0){
            s_temp[0] = s_qdd[2];
        }
        __syncthreads();
        grid_linalg_segmented_row_strided_gemv<T,6,6,6,true>(1, seg_a_off_lvl2, seg_a_x_off_lvl2, seg_a_y_off_lvl2, s_XImats, s_vaf, s_vaf, static_cast<T>(1), static_cast<T>(0), seg_s_off_lvl2, S_sel_lvl2, s_temp, s_linalg_smem);
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[54], &s_vaf[12], (1) * s_qd[2]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 3
        //     joints are: A4
        //     links are: L4
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + S[k]*qdd[k] + mxS[k](v[k])*qd[k]
        static const int seg_a_off_lvl3[1] = {108};
        static const int seg_v_x_off_lvl3[1] = {12};
        static const int seg_v_y_off_lvl3[1] = {18};
        static const int seg_s_off_lvl3[1] = {0};
        static const T S_sel_lvl3[6] = {static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0)};
        if(threadIdx.x == 0 && threadIdx.y == 0){
            s_temp[0] = s_qd[3];
        }
        __syncthreads();
        grid_linalg_segmented_row_strided_gemv<T,6,6,6,true>(1, seg_a_off_lvl3, seg_v_x_off_lvl3, seg_v_y_off_lvl3, s_XImats, s_vaf, s_vaf, static_cast<T>(1), static_cast<T>(0), seg_s_off_lvl3, S_sel_lvl3, s_temp, s_linalg_smem);
        static const int seg_a_x_off_lvl3[1] = {54};
        static const int seg_a_y_off_lvl3[1] = {60};
        if(threadIdx.x == 0 && threadIdx.y == 0){
            s_temp[0] = s_qdd[3];
        }
        __syncthreads();
        grid_linalg_segmented_row_strided_gemv<T,6,6,6,true>(1, seg_a_off_lvl3, seg_a_x_off_lvl3, seg_a_y_off_lvl3, s_XImats, s_vaf, s_vaf, static_cast<T>(1), static_cast<T>(0), seg_s_off_lvl3, S_sel_lvl3, s_temp, s_linalg_smem);
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[60], &s_vaf[18], (1) * s_qd[3]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 4
        //     joints are: A5
        //     links are: L5
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + S[k]*qdd[k] + mxS[k](v[k])*qd[k]
        static const int seg_a_off_lvl4[1] = {144};
        static const int seg_v_x_off_lvl4[1] = {18};
        static const int seg_v_y_off_lvl4[1] = {24};
        static const int seg_s_off_lvl4[1] = {0};
        static const T S_sel_lvl4[6] = {static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0)};
        if(threadIdx.x == 0 && threadIdx.y == 0){
            s_temp[0] = s_qd[4];
        }
        __syncthreads();
        grid_linalg_segmented_row_strided_gemv<T,6,6,6,true>(1, seg_a_off_lvl4, seg_v_x_off_lvl4, seg_v_y_off_lvl4, s_XImats, s_vaf, s_vaf, static_cast<T>(1), static_cast<T>(0), seg_s_off_lvl4, S_sel_lvl4, s_temp, s_linalg_smem);
        static const int seg_a_x_off_lvl4[1] = {60};
        static const int seg_a_y_off_lvl4[1] = {66};
        if(threadIdx.x == 0 && threadIdx.y == 0){
            s_temp[0] = s_qdd[4];
        }
        __syncthreads();
        grid_linalg_segmented_row_strided_gemv<T,6,6,6,true>(1, seg_a_off_lvl4, seg_a_x_off_lvl4, seg_a_y_off_lvl4, s_XImats, s_vaf, s_vaf, static_cast<T>(1), static_cast<T>(0), seg_s_off_lvl4, S_sel_lvl4, s_temp, s_linalg_smem);
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[66], &s_vaf[24], (1) * s_qd[4]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 5
        //     joints are: A6
        //     links are: L6
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + S[k]*qdd[k] + mxS[k](v[k])*qd[k]
        static const int seg_a_off_lvl5[1] = {180};
        static const int seg_v_x_off_lvl5[1] = {24};
        static const int seg_v_y_off_lvl5[1] = {30};
        static const int seg_s_off_lvl5[1] = {0};
        static const T S_sel_lvl5[6] = {static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0)};
        if(threadIdx.x == 0 && threadIdx.y == 0){
            s_temp[0] = s_qd[5];
        }
        __syncthreads();
        grid_linalg_segmented_row_strided_gemv<T,6,6,6,true>(1, seg_a_off_lvl5, seg_v_x_off_lvl5, seg_v_y_off_lvl5, s_XImats, s_vaf, s_vaf, static_cast<T>(1), static_cast<T>(0), seg_s_off_lvl5, S_sel_lvl5, s_temp, s_linalg_smem);
        static const int seg_a_x_off_lvl5[1] = {66};
        static const int seg_a_y_off_lvl5[1] = {72};
        if(threadIdx.x == 0 && threadIdx.y == 0){
            s_temp[0] = s_qdd[5];
        }
        __syncthreads();
        grid_linalg_segmented_row_strided_gemv<T,6,6,6,true>(1, seg_a_off_lvl5, seg_a_x_off_lvl5, seg_a_y_off_lvl5, s_XImats, s_vaf, s_vaf, static_cast<T>(1), static_cast<T>(0), seg_s_off_lvl5, S_sel_lvl5, s_temp, s_linalg_smem);
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[72], &s_vaf[30], (1) * s_qd[5]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 6
        //     joints are: A7
        //     links are: L7
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + S[k]*qdd[k] + mxS[k](v[k])*qd[k]
        static const int seg_a_off_lvl6[1] = {216};
        static const int seg_v_x_off_lvl6[1] = {30};
        static const int seg_v_y_off_lvl6[1] = {36};
        static const int seg_s_off_lvl6[1] = {0};
        static const T S_sel_lvl6[6] = {static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0)};
        if(threadIdx.x == 0 && threadIdx.y == 0){
            s_temp[0] = s_qd[6];
        }
        __syncthreads();
        grid_linalg_segmented_row_strided_gemv<T,6,6,6,true>(1, seg_a_off_lvl6, seg_v_x_off_lvl6, seg_v_y_off_lvl6, s_XImats, s_vaf, s_vaf, static_cast<T>(1), static_cast<T>(0), seg_s_off_lvl6, S_sel_lvl6, s_temp, s_linalg_smem);
        static const int seg_a_x_off_lvl6[1] = {72};
        static const int seg_a_y_off_lvl6[1] = {78};
        if(threadIdx.x == 0 && threadIdx.y == 0){
            s_temp[0] = s_qdd[6];
        }
        __syncthreads();
        grid_linalg_segmented_row_strided_gemv<T,6,6,6,true>(1, seg_a_off_lvl6, seg_a_x_off_lvl6, seg_a_y_off_lvl6, s_XImats, s_vaf, s_vaf, static_cast<T>(1), static_cast<T>(0), seg_s_off_lvl6, S_sel_lvl6, s_temp, s_linalg_smem);
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[78], &s_vaf[36], (1) * s_qd[6]);
        }
        __syncthreads();
        //
        // s_f in parallel given all v, a
        //
        // s_f[k] = I[k]*a[k] + fx(v[k])*I[k]*v[k]
        // start with s_f[k] = I[k]*a[k] and temp = *I[k]*v[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 84; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int jid = comp % 7;
            bool IaFlag = comp == jid; int jid6 = 6*jid; int vaOffset = IaFlag * 42 + jid6;
            T *dst = IaFlag ? &s_vaf[84] : s_temp;
            // compute based on the branch and save Iv to temp to prep for fx(v)*Iv and then sync
            dst[jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[252 + 6*jid6 + row], &s_vaf[vaOffset]);
        }
        __syncthreads();
        // finish with s_f[k] += fx(v[k])*Iv[k]
        for(int jid = threadIdx.x + threadIdx.y*blockDim.x; jid < 7; jid += blockDim.x*blockDim.y){
            int jid6 = 6*jid;
            fx_times_v_peq<T>(&s_vaf[84 + jid6], &s_vaf[jid6], &s_temp[jid6]);
            if (d_f_ext != nullptr) {
                for (int r = 0; r < 6; r++) { s_vaf[84 + jid6 + r] -= d_f_ext[jid6 + r]; }
            }
        }
        __syncthreads();
        //
        // Backward Pass
        //
        // s_f update where bfs_level is 6
        //     joints are: A7
        //     links are: L7
        // s_f[parent_k] += X[k]^T*f[k]
        grid_linalg_gemv<T,6,6,true>(&s_XImats[216], &s_vaf[120], &s_vaf[114], static_cast<T>(1), static_cast<T>(1));
        // s_f update where bfs_level is 5
        //     joints are: A6
        //     links are: L6
        // s_f[parent_k] += X[k]^T*f[k]
        grid_linalg_gemv<T,6,6,true>(&s_XImats[180], &s_vaf[114], &s_vaf[108], static_cast<T>(1), static_cast<T>(1));
        // s_f update where bfs_level is 4
        //     joints are: A5
        //     links are: L5
        // s_f[parent_k] += X[k]^T*f[k]
        grid_linalg_gemv<T,6,6,true>(&s_XImats[144], &s_vaf[108], &s_vaf[102], static_cast<T>(1), static_cast<T>(1));
        // s_f update where bfs_level is 3
        //     joints are: A4
        //     links are: L4
        // s_f[parent_k] += X[k]^T*f[k]
        grid_linalg_gemv<T,6,6,true>(&s_XImats[108], &s_vaf[102], &s_vaf[96], static_cast<T>(1), static_cast<T>(1));
        // s_f update where bfs_level is 2
        //     joints are: A3
        //     links are: L3
        // s_f[parent_k] += X[k]^T*f[k]
        grid_linalg_gemv<T,6,6,true>(&s_XImats[72], &s_vaf[96], &s_vaf[90], static_cast<T>(1), static_cast<T>(1));
        // s_f update where bfs_level is 1
        //     joints are: A2
        //     links are: L2
        // s_f[parent_k] += X[k]^T*f[k]
        grid_linalg_gemv<T,6,6,true>(&s_XImats[36], &s_vaf[90], &s_vaf[84], static_cast<T>(1), static_cast<T>(1));
        //
        // s_c extracted in parallel (S*f)
        //
        for(int dof_id = threadIdx.x + threadIdx.y*blockDim.x; dof_id < 7; dof_id += blockDim.x*blockDim.y){
            s_c[dof_id] = (1) * s_vaf[84 + 6*dof_id + 2];
        }
        __syncthreads();
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * Notes:
     *   Assumes the XI matricies have already been updated for the given q
     *   optimized for qdd = 0
     *
     * @param s_c is the vector of output torques
     * @param s_vaf is a pointer to shared memory of size 3*6*NUM_JOINTS = 126
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param s_XI is the pointer to the transformation and inertia matricies 
     * @param s_temp is a pointer to helper shared memory of size 6*NUM_JOINTS = 42
     * @param s_XImats is the (shared) memory holding the updated XI matricies for the given s_q
     * @param s_topology_helpers is the (shared) memory location for the topology_helpers (nullptr/unused for serial chains with identical Ss)
     * @param gravity is the gravity constant
     * @param d_f_ext is the (optional) GLOBAL external forces, body-major 6*NUM_BODIES local-frame, or nullptr
     */
    template <typename T>
    __device__
    void inverse_dynamics_inner(T *s_c,  T *s_vaf, const T *s_q, const T *s_qd, T *s_XImats, int *s_topology_helpers, T *s_temp, T *d_f_ext, const T gravity) {
        unsigned char *s_linalg_smem = nullptr;
        //
        // Forward Pass
        //
        // s_v, s_a where parent is base
        //     joints are: A1
        //     links are: L1
        // s_v[k] = S[k]*qd[k] and s_a[k] = X[k]*gravity
        for(int row = threadIdx.x + threadIdx.y*blockDim.x; row < 6; row += blockDim.x*blockDim.y){
            int jid6 = 6*0;
            s_vaf[jid6 + row] = static_cast<T>(0);
            s_vaf[42 + jid6 + row] = -s_XImats[6*jid6 + 30 + row]*gravity;
            if (row == 2){s_vaf[jid6 + 2] += (1) * s_qd[0];}
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 1
        //     joints are: A2
        //     links are: L2
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + mxS[k](v[k])*qd[k]
        static const int seg_a_off_lvl1[1] = {36};
        static const int seg_v_x_off_lvl1[1] = {0};
        static const int seg_v_y_off_lvl1[1] = {6};
        static const int seg_s_off_lvl1[1] = {0};
        static const T S_sel_lvl1[6] = {static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0)};
        if(threadIdx.x == 0 && threadIdx.y == 0){
            s_temp[0] = s_qd[1];
        }
        __syncthreads();
        grid_linalg_segmented_row_strided_gemv<T,6,6,6,true>(1, seg_a_off_lvl1, seg_v_x_off_lvl1, seg_v_y_off_lvl1, s_XImats, s_vaf, s_vaf, static_cast<T>(1), static_cast<T>(0), seg_s_off_lvl1, S_sel_lvl1, s_temp, s_linalg_smem);
        static const int seg_a_x_off_lvl1[1] = {42};
        static const int seg_a_y_off_lvl1[1] = {48};
        grid_linalg_segmented_row_strided_gemv<T,6,6,6>(1, seg_a_off_lvl1, seg_a_x_off_lvl1, seg_a_y_off_lvl1, s_XImats, s_vaf, s_vaf, static_cast<T>(1), static_cast<T>(0));
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[48], &s_vaf[6], (1) * s_qd[1]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 2
        //     joints are: A3
        //     links are: L3
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + mxS[k](v[k])*qd[k]
        static const int seg_a_off_lvl2[1] = {72};
        static const int seg_v_x_off_lvl2[1] = {6};
        static const int seg_v_y_off_lvl2[1] = {12};
        static const int seg_s_off_lvl2[1] = {0};
        static const T S_sel_lvl2[6] = {static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0)};
        if(threadIdx.x == 0 && threadIdx.y == 0){
            s_temp[0] = s_qd[2];
        }
        __syncthreads();
        grid_linalg_segmented_row_strided_gemv<T,6,6,6,true>(1, seg_a_off_lvl2, seg_v_x_off_lvl2, seg_v_y_off_lvl2, s_XImats, s_vaf, s_vaf, static_cast<T>(1), static_cast<T>(0), seg_s_off_lvl2, S_sel_lvl2, s_temp, s_linalg_smem);
        static const int seg_a_x_off_lvl2[1] = {48};
        static const int seg_a_y_off_lvl2[1] = {54};
        grid_linalg_segmented_row_strided_gemv<T,6,6,6>(1, seg_a_off_lvl2, seg_a_x_off_lvl2, seg_a_y_off_lvl2, s_XImats, s_vaf, s_vaf, static_cast<T>(1), static_cast<T>(0));
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[54], &s_vaf[12], (1) * s_qd[2]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 3
        //     joints are: A4
        //     links are: L4
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + mxS[k](v[k])*qd[k]
        static const int seg_a_off_lvl3[1] = {108};
        static const int seg_v_x_off_lvl3[1] = {12};
        static const int seg_v_y_off_lvl3[1] = {18};
        static const int seg_s_off_lvl3[1] = {0};
        static const T S_sel_lvl3[6] = {static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0)};
        if(threadIdx.x == 0 && threadIdx.y == 0){
            s_temp[0] = s_qd[3];
        }
        __syncthreads();
        grid_linalg_segmented_row_strided_gemv<T,6,6,6,true>(1, seg_a_off_lvl3, seg_v_x_off_lvl3, seg_v_y_off_lvl3, s_XImats, s_vaf, s_vaf, static_cast<T>(1), static_cast<T>(0), seg_s_off_lvl3, S_sel_lvl3, s_temp, s_linalg_smem);
        static const int seg_a_x_off_lvl3[1] = {54};
        static const int seg_a_y_off_lvl3[1] = {60};
        grid_linalg_segmented_row_strided_gemv<T,6,6,6>(1, seg_a_off_lvl3, seg_a_x_off_lvl3, seg_a_y_off_lvl3, s_XImats, s_vaf, s_vaf, static_cast<T>(1), static_cast<T>(0));
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[60], &s_vaf[18], (1) * s_qd[3]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 4
        //     joints are: A5
        //     links are: L5
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + mxS[k](v[k])*qd[k]
        static const int seg_a_off_lvl4[1] = {144};
        static const int seg_v_x_off_lvl4[1] = {18};
        static const int seg_v_y_off_lvl4[1] = {24};
        static const int seg_s_off_lvl4[1] = {0};
        static const T S_sel_lvl4[6] = {static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0)};
        if(threadIdx.x == 0 && threadIdx.y == 0){
            s_temp[0] = s_qd[4];
        }
        __syncthreads();
        grid_linalg_segmented_row_strided_gemv<T,6,6,6,true>(1, seg_a_off_lvl4, seg_v_x_off_lvl4, seg_v_y_off_lvl4, s_XImats, s_vaf, s_vaf, static_cast<T>(1), static_cast<T>(0), seg_s_off_lvl4, S_sel_lvl4, s_temp, s_linalg_smem);
        static const int seg_a_x_off_lvl4[1] = {60};
        static const int seg_a_y_off_lvl4[1] = {66};
        grid_linalg_segmented_row_strided_gemv<T,6,6,6>(1, seg_a_off_lvl4, seg_a_x_off_lvl4, seg_a_y_off_lvl4, s_XImats, s_vaf, s_vaf, static_cast<T>(1), static_cast<T>(0));
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[66], &s_vaf[24], (1) * s_qd[4]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 5
        //     joints are: A6
        //     links are: L6
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + mxS[k](v[k])*qd[k]
        static const int seg_a_off_lvl5[1] = {180};
        static const int seg_v_x_off_lvl5[1] = {24};
        static const int seg_v_y_off_lvl5[1] = {30};
        static const int seg_s_off_lvl5[1] = {0};
        static const T S_sel_lvl5[6] = {static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0)};
        if(threadIdx.x == 0 && threadIdx.y == 0){
            s_temp[0] = s_qd[5];
        }
        __syncthreads();
        grid_linalg_segmented_row_strided_gemv<T,6,6,6,true>(1, seg_a_off_lvl5, seg_v_x_off_lvl5, seg_v_y_off_lvl5, s_XImats, s_vaf, s_vaf, static_cast<T>(1), static_cast<T>(0), seg_s_off_lvl5, S_sel_lvl5, s_temp, s_linalg_smem);
        static const int seg_a_x_off_lvl5[1] = {66};
        static const int seg_a_y_off_lvl5[1] = {72};
        grid_linalg_segmented_row_strided_gemv<T,6,6,6>(1, seg_a_off_lvl5, seg_a_x_off_lvl5, seg_a_y_off_lvl5, s_XImats, s_vaf, s_vaf, static_cast<T>(1), static_cast<T>(0));
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[72], &s_vaf[30], (1) * s_qd[5]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 6
        //     joints are: A7
        //     links are: L7
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + mxS[k](v[k])*qd[k]
        static const int seg_a_off_lvl6[1] = {216};
        static const int seg_v_x_off_lvl6[1] = {30};
        static const int seg_v_y_off_lvl6[1] = {36};
        static const int seg_s_off_lvl6[1] = {0};
        static const T S_sel_lvl6[6] = {static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0)};
        if(threadIdx.x == 0 && threadIdx.y == 0){
            s_temp[0] = s_qd[6];
        }
        __syncthreads();
        grid_linalg_segmented_row_strided_gemv<T,6,6,6,true>(1, seg_a_off_lvl6, seg_v_x_off_lvl6, seg_v_y_off_lvl6, s_XImats, s_vaf, s_vaf, static_cast<T>(1), static_cast<T>(0), seg_s_off_lvl6, S_sel_lvl6, s_temp, s_linalg_smem);
        static const int seg_a_x_off_lvl6[1] = {72};
        static const int seg_a_y_off_lvl6[1] = {78};
        grid_linalg_segmented_row_strided_gemv<T,6,6,6>(1, seg_a_off_lvl6, seg_a_x_off_lvl6, seg_a_y_off_lvl6, s_XImats, s_vaf, s_vaf, static_cast<T>(1), static_cast<T>(0));
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[78], &s_vaf[36], (1) * s_qd[6]);
        }
        __syncthreads();
        //
        // s_f in parallel given all v, a
        //
        // s_f[k] = I[k]*a[k] + fx(v[k])*I[k]*v[k]
        // start with s_f[k] = I[k]*a[k] and temp = *I[k]*v[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 84; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int jid = comp % 7;
            bool IaFlag = comp == jid; int jid6 = 6*jid; int vaOffset = IaFlag * 42 + jid6;
            T *dst = IaFlag ? &s_vaf[84] : s_temp;
            // compute based on the branch and save Iv to temp to prep for fx(v)*Iv and then sync
            dst[jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[252 + 6*jid6 + row], &s_vaf[vaOffset]);
        }
        __syncthreads();
        // finish with s_f[k] += fx(v[k])*Iv[k]
        for(int jid = threadIdx.x + threadIdx.y*blockDim.x; jid < 7; jid += blockDim.x*blockDim.y){
            int jid6 = 6*jid;
            fx_times_v_peq<T>(&s_vaf[84 + jid6], &s_vaf[jid6], &s_temp[jid6]);
            if (d_f_ext != nullptr) {
                for (int r = 0; r < 6; r++) { s_vaf[84 + jid6 + r] -= d_f_ext[jid6 + r]; }
            }
        }
        __syncthreads();
        //
        // Backward Pass
        //
        // s_f update where bfs_level is 6
        //     joints are: A7
        //     links are: L7
        // s_f[parent_k] += X[k]^T*f[k]
        grid_linalg_gemv<T,6,6,true>(&s_XImats[216], &s_vaf[120], &s_vaf[114], static_cast<T>(1), static_cast<T>(1));
        // s_f update where bfs_level is 5
        //     joints are: A6
        //     links are: L6
        // s_f[parent_k] += X[k]^T*f[k]
        grid_linalg_gemv<T,6,6,true>(&s_XImats[180], &s_vaf[114], &s_vaf[108], static_cast<T>(1), static_cast<T>(1));
        // s_f update where bfs_level is 4
        //     joints are: A5
        //     links are: L5
        // s_f[parent_k] += X[k]^T*f[k]
        grid_linalg_gemv<T,6,6,true>(&s_XImats[144], &s_vaf[108], &s_vaf[102], static_cast<T>(1), static_cast<T>(1));
        // s_f update where bfs_level is 3
        //     joints are: A4
        //     links are: L4
        // s_f[parent_k] += X[k]^T*f[k]
        grid_linalg_gemv<T,6,6,true>(&s_XImats[108], &s_vaf[102], &s_vaf[96], static_cast<T>(1), static_cast<T>(1));
        // s_f update where bfs_level is 2
        //     joints are: A3
        //     links are: L3
        // s_f[parent_k] += X[k]^T*f[k]
        grid_linalg_gemv<T,6,6,true>(&s_XImats[72], &s_vaf[96], &s_vaf[90], static_cast<T>(1), static_cast<T>(1));
        // s_f update where bfs_level is 1
        //     joints are: A2
        //     links are: L2
        // s_f[parent_k] += X[k]^T*f[k]
        grid_linalg_gemv<T,6,6,true>(&s_XImats[36], &s_vaf[90], &s_vaf[84], static_cast<T>(1), static_cast<T>(1));
        //
        // s_c extracted in parallel (S*f)
        //
        for(int dof_id = threadIdx.x + threadIdx.y*blockDim.x; dof_id < 7; dof_id += blockDim.x*blockDim.y){
            s_c[dof_id] = (1) * s_vaf[84 + 6*dof_id + 2];
        }
        __syncthreads();
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * Notes:
     *   Assumes the XI matricies have already been updated for the given q
     *   used to compute vaf as helper values
     *
     * @param s_vaf is a pointer to shared memory of size 3*6*NUM_JOINTS = 126
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param s_XI is the pointer to the transformation and inertia matricies 
     * @param s_qdd is (optional vector of joint accelerations
     * @param s_temp is a pointer to helper shared memory of size 6*NUM_JOINTS = 42
     * @param s_XImats is the (shared) memory holding the updated XI matricies for the given s_q
     * @param s_topology_helpers is the (shared) memory location for the topology_helpers (nullptr/unused for serial chains with identical Ss)
     * @param gravity is the gravity constant
     * @param d_f_ext is the (optional) GLOBAL external forces, body-major 6*NUM_BODIES local-frame, or nullptr
     */
    template <typename T>
    __device__
    void inverse_dynamics_inner_vaf(T *s_vaf, const T *s_q, const T *s_qd, const T *s_qdd, T *s_XImats, int *s_topology_helpers, T *s_temp, T *d_f_ext, const T gravity) {
        unsigned char *s_linalg_smem = nullptr;
        //
        // Forward Pass
        //
        // s_v, s_a where parent is base
        //     joints are: A1
        //     links are: L1
        // s_v[k] = S[k]*qd[k] and s_a[k] = X[k]*gravityS[k]*qdd[k]
        for(int row = threadIdx.x + threadIdx.y*blockDim.x; row < 6; row += blockDim.x*blockDim.y){
            int jid6 = 6*0;
            s_vaf[jid6 + row] = static_cast<T>(0);
            s_vaf[42 + jid6 + row] = -s_XImats[6*jid6 + 30 + row]*gravity;
            if (row == 2){s_vaf[jid6 + 2] += (1) * s_qd[0]; s_vaf[42 + jid6 + 2] += (1) * s_qdd[0];}
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 1
        //     joints are: A2
        //     links are: L2
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + S[k]*qdd[k] + mxS[k](v[k])*qd[k]
        static const int seg_a_off_lvl1[1] = {36};
        static const int seg_v_x_off_lvl1[1] = {0};
        static const int seg_v_y_off_lvl1[1] = {6};
        static const int seg_s_off_lvl1[1] = {0};
        static const T S_sel_lvl1[6] = {static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0)};
        if(threadIdx.x == 0 && threadIdx.y == 0){
            s_temp[0] = s_qd[1];
        }
        __syncthreads();
        grid_linalg_segmented_row_strided_gemv<T,6,6,6,true>(1, seg_a_off_lvl1, seg_v_x_off_lvl1, seg_v_y_off_lvl1, s_XImats, s_vaf, s_vaf, static_cast<T>(1), static_cast<T>(0), seg_s_off_lvl1, S_sel_lvl1, s_temp, s_linalg_smem);
        static const int seg_a_x_off_lvl1[1] = {42};
        static const int seg_a_y_off_lvl1[1] = {48};
        if(threadIdx.x == 0 && threadIdx.y == 0){
            s_temp[0] = s_qdd[1];
        }
        __syncthreads();
        grid_linalg_segmented_row_strided_gemv<T,6,6,6,true>(1, seg_a_off_lvl1, seg_a_x_off_lvl1, seg_a_y_off_lvl1, s_XImats, s_vaf, s_vaf, static_cast<T>(1), static_cast<T>(0), seg_s_off_lvl1, S_sel_lvl1, s_temp, s_linalg_smem);
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[48], &s_vaf[6], (1) * s_qd[1]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 2
        //     joints are: A3
        //     links are: L3
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + S[k]*qdd[k] + mxS[k](v[k])*qd[k]
        static const int seg_a_off_lvl2[1] = {72};
        static const int seg_v_x_off_lvl2[1] = {6};
        static const int seg_v_y_off_lvl2[1] = {12};
        static const int seg_s_off_lvl2[1] = {0};
        static const T S_sel_lvl2[6] = {static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0)};
        if(threadIdx.x == 0 && threadIdx.y == 0){
            s_temp[0] = s_qd[2];
        }
        __syncthreads();
        grid_linalg_segmented_row_strided_gemv<T,6,6,6,true>(1, seg_a_off_lvl2, seg_v_x_off_lvl2, seg_v_y_off_lvl2, s_XImats, s_vaf, s_vaf, static_cast<T>(1), static_cast<T>(0), seg_s_off_lvl2, S_sel_lvl2, s_temp, s_linalg_smem);
        static const int seg_a_x_off_lvl2[1] = {48};
        static const int seg_a_y_off_lvl2[1] = {54};
        if(threadIdx.x == 0 && threadIdx.y == 0){
            s_temp[0] = s_qdd[2];
        }
        __syncthreads();
        grid_linalg_segmented_row_strided_gemv<T,6,6,6,true>(1, seg_a_off_lvl2, seg_a_x_off_lvl2, seg_a_y_off_lvl2, s_XImats, s_vaf, s_vaf, static_cast<T>(1), static_cast<T>(0), seg_s_off_lvl2, S_sel_lvl2, s_temp, s_linalg_smem);
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[54], &s_vaf[12], (1) * s_qd[2]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 3
        //     joints are: A4
        //     links are: L4
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + S[k]*qdd[k] + mxS[k](v[k])*qd[k]
        static const int seg_a_off_lvl3[1] = {108};
        static const int seg_v_x_off_lvl3[1] = {12};
        static const int seg_v_y_off_lvl3[1] = {18};
        static const int seg_s_off_lvl3[1] = {0};
        static const T S_sel_lvl3[6] = {static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0)};
        if(threadIdx.x == 0 && threadIdx.y == 0){
            s_temp[0] = s_qd[3];
        }
        __syncthreads();
        grid_linalg_segmented_row_strided_gemv<T,6,6,6,true>(1, seg_a_off_lvl3, seg_v_x_off_lvl3, seg_v_y_off_lvl3, s_XImats, s_vaf, s_vaf, static_cast<T>(1), static_cast<T>(0), seg_s_off_lvl3, S_sel_lvl3, s_temp, s_linalg_smem);
        static const int seg_a_x_off_lvl3[1] = {54};
        static const int seg_a_y_off_lvl3[1] = {60};
        if(threadIdx.x == 0 && threadIdx.y == 0){
            s_temp[0] = s_qdd[3];
        }
        __syncthreads();
        grid_linalg_segmented_row_strided_gemv<T,6,6,6,true>(1, seg_a_off_lvl3, seg_a_x_off_lvl3, seg_a_y_off_lvl3, s_XImats, s_vaf, s_vaf, static_cast<T>(1), static_cast<T>(0), seg_s_off_lvl3, S_sel_lvl3, s_temp, s_linalg_smem);
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[60], &s_vaf[18], (1) * s_qd[3]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 4
        //     joints are: A5
        //     links are: L5
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + S[k]*qdd[k] + mxS[k](v[k])*qd[k]
        static const int seg_a_off_lvl4[1] = {144};
        static const int seg_v_x_off_lvl4[1] = {18};
        static const int seg_v_y_off_lvl4[1] = {24};
        static const int seg_s_off_lvl4[1] = {0};
        static const T S_sel_lvl4[6] = {static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0)};
        if(threadIdx.x == 0 && threadIdx.y == 0){
            s_temp[0] = s_qd[4];
        }
        __syncthreads();
        grid_linalg_segmented_row_strided_gemv<T,6,6,6,true>(1, seg_a_off_lvl4, seg_v_x_off_lvl4, seg_v_y_off_lvl4, s_XImats, s_vaf, s_vaf, static_cast<T>(1), static_cast<T>(0), seg_s_off_lvl4, S_sel_lvl4, s_temp, s_linalg_smem);
        static const int seg_a_x_off_lvl4[1] = {60};
        static const int seg_a_y_off_lvl4[1] = {66};
        if(threadIdx.x == 0 && threadIdx.y == 0){
            s_temp[0] = s_qdd[4];
        }
        __syncthreads();
        grid_linalg_segmented_row_strided_gemv<T,6,6,6,true>(1, seg_a_off_lvl4, seg_a_x_off_lvl4, seg_a_y_off_lvl4, s_XImats, s_vaf, s_vaf, static_cast<T>(1), static_cast<T>(0), seg_s_off_lvl4, S_sel_lvl4, s_temp, s_linalg_smem);
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[66], &s_vaf[24], (1) * s_qd[4]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 5
        //     joints are: A6
        //     links are: L6
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + S[k]*qdd[k] + mxS[k](v[k])*qd[k]
        static const int seg_a_off_lvl5[1] = {180};
        static const int seg_v_x_off_lvl5[1] = {24};
        static const int seg_v_y_off_lvl5[1] = {30};
        static const int seg_s_off_lvl5[1] = {0};
        static const T S_sel_lvl5[6] = {static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0)};
        if(threadIdx.x == 0 && threadIdx.y == 0){
            s_temp[0] = s_qd[5];
        }
        __syncthreads();
        grid_linalg_segmented_row_strided_gemv<T,6,6,6,true>(1, seg_a_off_lvl5, seg_v_x_off_lvl5, seg_v_y_off_lvl5, s_XImats, s_vaf, s_vaf, static_cast<T>(1), static_cast<T>(0), seg_s_off_lvl5, S_sel_lvl5, s_temp, s_linalg_smem);
        static const int seg_a_x_off_lvl5[1] = {66};
        static const int seg_a_y_off_lvl5[1] = {72};
        if(threadIdx.x == 0 && threadIdx.y == 0){
            s_temp[0] = s_qdd[5];
        }
        __syncthreads();
        grid_linalg_segmented_row_strided_gemv<T,6,6,6,true>(1, seg_a_off_lvl5, seg_a_x_off_lvl5, seg_a_y_off_lvl5, s_XImats, s_vaf, s_vaf, static_cast<T>(1), static_cast<T>(0), seg_s_off_lvl5, S_sel_lvl5, s_temp, s_linalg_smem);
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[72], &s_vaf[30], (1) * s_qd[5]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 6
        //     joints are: A7
        //     links are: L7
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + S[k]*qdd[k] + mxS[k](v[k])*qd[k]
        static const int seg_a_off_lvl6[1] = {216};
        static const int seg_v_x_off_lvl6[1] = {30};
        static const int seg_v_y_off_lvl6[1] = {36};
        static const int seg_s_off_lvl6[1] = {0};
        static const T S_sel_lvl6[6] = {static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0)};
        if(threadIdx.x == 0 && threadIdx.y == 0){
            s_temp[0] = s_qd[6];
        }
        __syncthreads();
        grid_linalg_segmented_row_strided_gemv<T,6,6,6,true>(1, seg_a_off_lvl6, seg_v_x_off_lvl6, seg_v_y_off_lvl6, s_XImats, s_vaf, s_vaf, static_cast<T>(1), static_cast<T>(0), seg_s_off_lvl6, S_sel_lvl6, s_temp, s_linalg_smem);
        static const int seg_a_x_off_lvl6[1] = {72};
        static const int seg_a_y_off_lvl6[1] = {78};
        if(threadIdx.x == 0 && threadIdx.y == 0){
            s_temp[0] = s_qdd[6];
        }
        __syncthreads();
        grid_linalg_segmented_row_strided_gemv<T,6,6,6,true>(1, seg_a_off_lvl6, seg_a_x_off_lvl6, seg_a_y_off_lvl6, s_XImats, s_vaf, s_vaf, static_cast<T>(1), static_cast<T>(0), seg_s_off_lvl6, S_sel_lvl6, s_temp, s_linalg_smem);
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[78], &s_vaf[36], (1) * s_qd[6]);
        }
        __syncthreads();
        //
        // s_f in parallel given all v, a
        //
        // s_f[k] = I[k]*a[k] + fx(v[k])*I[k]*v[k]
        // start with s_f[k] = I[k]*a[k] and temp = *I[k]*v[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 84; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int jid = comp % 7;
            bool IaFlag = comp == jid; int jid6 = 6*jid; int vaOffset = IaFlag * 42 + jid6;
            T *dst = IaFlag ? &s_vaf[84] : s_temp;
            // compute based on the branch and save Iv to temp to prep for fx(v)*Iv and then sync
            dst[jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[252 + 6*jid6 + row], &s_vaf[vaOffset]);
        }
        __syncthreads();
        // finish with s_f[k] += fx(v[k])*Iv[k]
        for(int jid = threadIdx.x + threadIdx.y*blockDim.x; jid < 7; jid += blockDim.x*blockDim.y){
            int jid6 = 6*jid;
            fx_times_v_peq<T>(&s_vaf[84 + jid6], &s_vaf[jid6], &s_temp[jid6]);
            if (d_f_ext != nullptr) {
                for (int r = 0; r < 6; r++) { s_vaf[84 + jid6 + r] -= d_f_ext[jid6 + r]; }
            }
        }
        __syncthreads();
        //
        // Backward Pass
        //
        // s_f update where bfs_level is 6
        //     joints are: A7
        //     links are: L7
        // s_f[parent_k] += X[k]^T*f[k]
        grid_linalg_gemv<T,6,6,true>(&s_XImats[216], &s_vaf[120], &s_vaf[114], static_cast<T>(1), static_cast<T>(1));
        // s_f update where bfs_level is 5
        //     joints are: A6
        //     links are: L6
        // s_f[parent_k] += X[k]^T*f[k]
        grid_linalg_gemv<T,6,6,true>(&s_XImats[180], &s_vaf[114], &s_vaf[108], static_cast<T>(1), static_cast<T>(1));
        // s_f update where bfs_level is 4
        //     joints are: A5
        //     links are: L5
        // s_f[parent_k] += X[k]^T*f[k]
        grid_linalg_gemv<T,6,6,true>(&s_XImats[144], &s_vaf[108], &s_vaf[102], static_cast<T>(1), static_cast<T>(1));
        // s_f update where bfs_level is 3
        //     joints are: A4
        //     links are: L4
        // s_f[parent_k] += X[k]^T*f[k]
        grid_linalg_gemv<T,6,6,true>(&s_XImats[108], &s_vaf[102], &s_vaf[96], static_cast<T>(1), static_cast<T>(1));
        // s_f update where bfs_level is 2
        //     joints are: A3
        //     links are: L3
        // s_f[parent_k] += X[k]^T*f[k]
        grid_linalg_gemv<T,6,6,true>(&s_XImats[72], &s_vaf[96], &s_vaf[90], static_cast<T>(1), static_cast<T>(1));
        // s_f update where bfs_level is 1
        //     joints are: A2
        //     links are: L2
        // s_f[parent_k] += X[k]^T*f[k]
        grid_linalg_gemv<T,6,6,true>(&s_XImats[36], &s_vaf[90], &s_vaf[84], static_cast<T>(1), static_cast<T>(1));
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * Notes:
     *   Assumes the XI matricies have already been updated for the given q
     *   used to compute vaf as helper values
     *   optimized for qdd = 0
     *
     * @param s_vaf is a pointer to shared memory of size 3*6*NUM_JOINTS = 126
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param s_XI is the pointer to the transformation and inertia matricies 
     * @param s_temp is a pointer to helper shared memory of size 6*NUM_JOINTS = 42
     * @param s_XImats is the (shared) memory holding the updated XI matricies for the given s_q
     * @param s_topology_helpers is the (shared) memory location for the topology_helpers (nullptr/unused for serial chains with identical Ss)
     * @param gravity is the gravity constant
     * @param d_f_ext is the (optional) GLOBAL external forces, body-major 6*NUM_BODIES local-frame, or nullptr
     */
    template <typename T>
    __device__
    void inverse_dynamics_inner_vaf(T *s_vaf, const T *s_q, const T *s_qd, T *s_XImats, int *s_topology_helpers, T *s_temp, T *d_f_ext, const T gravity) {
        unsigned char *s_linalg_smem = nullptr;
        //
        // Forward Pass
        //
        // s_v, s_a where parent is base
        //     joints are: A1
        //     links are: L1
        // s_v[k] = S[k]*qd[k] and s_a[k] = X[k]*gravity
        for(int row = threadIdx.x + threadIdx.y*blockDim.x; row < 6; row += blockDim.x*blockDim.y){
            int jid6 = 6*0;
            s_vaf[jid6 + row] = static_cast<T>(0);
            s_vaf[42 + jid6 + row] = -s_XImats[6*jid6 + 30 + row]*gravity;
            if (row == 2){s_vaf[jid6 + 2] += (1) * s_qd[0];}
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 1
        //     joints are: A2
        //     links are: L2
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + mxS[k](v[k])*qd[k]
        static const int seg_a_off_lvl1[1] = {36};
        static const int seg_v_x_off_lvl1[1] = {0};
        static const int seg_v_y_off_lvl1[1] = {6};
        static const int seg_s_off_lvl1[1] = {0};
        static const T S_sel_lvl1[6] = {static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0)};
        if(threadIdx.x == 0 && threadIdx.y == 0){
            s_temp[0] = s_qd[1];
        }
        __syncthreads();
        grid_linalg_segmented_row_strided_gemv<T,6,6,6,true>(1, seg_a_off_lvl1, seg_v_x_off_lvl1, seg_v_y_off_lvl1, s_XImats, s_vaf, s_vaf, static_cast<T>(1), static_cast<T>(0), seg_s_off_lvl1, S_sel_lvl1, s_temp, s_linalg_smem);
        static const int seg_a_x_off_lvl1[1] = {42};
        static const int seg_a_y_off_lvl1[1] = {48};
        grid_linalg_segmented_row_strided_gemv<T,6,6,6>(1, seg_a_off_lvl1, seg_a_x_off_lvl1, seg_a_y_off_lvl1, s_XImats, s_vaf, s_vaf, static_cast<T>(1), static_cast<T>(0));
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[48], &s_vaf[6], (1) * s_qd[1]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 2
        //     joints are: A3
        //     links are: L3
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + mxS[k](v[k])*qd[k]
        static const int seg_a_off_lvl2[1] = {72};
        static const int seg_v_x_off_lvl2[1] = {6};
        static const int seg_v_y_off_lvl2[1] = {12};
        static const int seg_s_off_lvl2[1] = {0};
        static const T S_sel_lvl2[6] = {static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0)};
        if(threadIdx.x == 0 && threadIdx.y == 0){
            s_temp[0] = s_qd[2];
        }
        __syncthreads();
        grid_linalg_segmented_row_strided_gemv<T,6,6,6,true>(1, seg_a_off_lvl2, seg_v_x_off_lvl2, seg_v_y_off_lvl2, s_XImats, s_vaf, s_vaf, static_cast<T>(1), static_cast<T>(0), seg_s_off_lvl2, S_sel_lvl2, s_temp, s_linalg_smem);
        static const int seg_a_x_off_lvl2[1] = {48};
        static const int seg_a_y_off_lvl2[1] = {54};
        grid_linalg_segmented_row_strided_gemv<T,6,6,6>(1, seg_a_off_lvl2, seg_a_x_off_lvl2, seg_a_y_off_lvl2, s_XImats, s_vaf, s_vaf, static_cast<T>(1), static_cast<T>(0));
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[54], &s_vaf[12], (1) * s_qd[2]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 3
        //     joints are: A4
        //     links are: L4
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + mxS[k](v[k])*qd[k]
        static const int seg_a_off_lvl3[1] = {108};
        static const int seg_v_x_off_lvl3[1] = {12};
        static const int seg_v_y_off_lvl3[1] = {18};
        static const int seg_s_off_lvl3[1] = {0};
        static const T S_sel_lvl3[6] = {static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0)};
        if(threadIdx.x == 0 && threadIdx.y == 0){
            s_temp[0] = s_qd[3];
        }
        __syncthreads();
        grid_linalg_segmented_row_strided_gemv<T,6,6,6,true>(1, seg_a_off_lvl3, seg_v_x_off_lvl3, seg_v_y_off_lvl3, s_XImats, s_vaf, s_vaf, static_cast<T>(1), static_cast<T>(0), seg_s_off_lvl3, S_sel_lvl3, s_temp, s_linalg_smem);
        static const int seg_a_x_off_lvl3[1] = {54};
        static const int seg_a_y_off_lvl3[1] = {60};
        grid_linalg_segmented_row_strided_gemv<T,6,6,6>(1, seg_a_off_lvl3, seg_a_x_off_lvl3, seg_a_y_off_lvl3, s_XImats, s_vaf, s_vaf, static_cast<T>(1), static_cast<T>(0));
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[60], &s_vaf[18], (1) * s_qd[3]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 4
        //     joints are: A5
        //     links are: L5
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + mxS[k](v[k])*qd[k]
        static const int seg_a_off_lvl4[1] = {144};
        static const int seg_v_x_off_lvl4[1] = {18};
        static const int seg_v_y_off_lvl4[1] = {24};
        static const int seg_s_off_lvl4[1] = {0};
        static const T S_sel_lvl4[6] = {static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0)};
        if(threadIdx.x == 0 && threadIdx.y == 0){
            s_temp[0] = s_qd[4];
        }
        __syncthreads();
        grid_linalg_segmented_row_strided_gemv<T,6,6,6,true>(1, seg_a_off_lvl4, seg_v_x_off_lvl4, seg_v_y_off_lvl4, s_XImats, s_vaf, s_vaf, static_cast<T>(1), static_cast<T>(0), seg_s_off_lvl4, S_sel_lvl4, s_temp, s_linalg_smem);
        static const int seg_a_x_off_lvl4[1] = {60};
        static const int seg_a_y_off_lvl4[1] = {66};
        grid_linalg_segmented_row_strided_gemv<T,6,6,6>(1, seg_a_off_lvl4, seg_a_x_off_lvl4, seg_a_y_off_lvl4, s_XImats, s_vaf, s_vaf, static_cast<T>(1), static_cast<T>(0));
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[66], &s_vaf[24], (1) * s_qd[4]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 5
        //     joints are: A6
        //     links are: L6
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + mxS[k](v[k])*qd[k]
        static const int seg_a_off_lvl5[1] = {180};
        static const int seg_v_x_off_lvl5[1] = {24};
        static const int seg_v_y_off_lvl5[1] = {30};
        static const int seg_s_off_lvl5[1] = {0};
        static const T S_sel_lvl5[6] = {static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0)};
        if(threadIdx.x == 0 && threadIdx.y == 0){
            s_temp[0] = s_qd[5];
        }
        __syncthreads();
        grid_linalg_segmented_row_strided_gemv<T,6,6,6,true>(1, seg_a_off_lvl5, seg_v_x_off_lvl5, seg_v_y_off_lvl5, s_XImats, s_vaf, s_vaf, static_cast<T>(1), static_cast<T>(0), seg_s_off_lvl5, S_sel_lvl5, s_temp, s_linalg_smem);
        static const int seg_a_x_off_lvl5[1] = {66};
        static const int seg_a_y_off_lvl5[1] = {72};
        grid_linalg_segmented_row_strided_gemv<T,6,6,6>(1, seg_a_off_lvl5, seg_a_x_off_lvl5, seg_a_y_off_lvl5, s_XImats, s_vaf, s_vaf, static_cast<T>(1), static_cast<T>(0));
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[72], &s_vaf[30], (1) * s_qd[5]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 6
        //     joints are: A7
        //     links are: L7
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + mxS[k](v[k])*qd[k]
        static const int seg_a_off_lvl6[1] = {216};
        static const int seg_v_x_off_lvl6[1] = {30};
        static const int seg_v_y_off_lvl6[1] = {36};
        static const int seg_s_off_lvl6[1] = {0};
        static const T S_sel_lvl6[6] = {static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0)};
        if(threadIdx.x == 0 && threadIdx.y == 0){
            s_temp[0] = s_qd[6];
        }
        __syncthreads();
        grid_linalg_segmented_row_strided_gemv<T,6,6,6,true>(1, seg_a_off_lvl6, seg_v_x_off_lvl6, seg_v_y_off_lvl6, s_XImats, s_vaf, s_vaf, static_cast<T>(1), static_cast<T>(0), seg_s_off_lvl6, S_sel_lvl6, s_temp, s_linalg_smem);
        static const int seg_a_x_off_lvl6[1] = {72};
        static const int seg_a_y_off_lvl6[1] = {78};
        grid_linalg_segmented_row_strided_gemv<T,6,6,6>(1, seg_a_off_lvl6, seg_a_x_off_lvl6, seg_a_y_off_lvl6, s_XImats, s_vaf, s_vaf, static_cast<T>(1), static_cast<T>(0));
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[78], &s_vaf[36], (1) * s_qd[6]);
        }
        __syncthreads();
        //
        // s_f in parallel given all v, a
        //
        // s_f[k] = I[k]*a[k] + fx(v[k])*I[k]*v[k]
        // start with s_f[k] = I[k]*a[k] and temp = *I[k]*v[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 84; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int jid = comp % 7;
            bool IaFlag = comp == jid; int jid6 = 6*jid; int vaOffset = IaFlag * 42 + jid6;
            T *dst = IaFlag ? &s_vaf[84] : s_temp;
            // compute based on the branch and save Iv to temp to prep for fx(v)*Iv and then sync
            dst[jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[252 + 6*jid6 + row], &s_vaf[vaOffset]);
        }
        __syncthreads();
        // finish with s_f[k] += fx(v[k])*Iv[k]
        for(int jid = threadIdx.x + threadIdx.y*blockDim.x; jid < 7; jid += blockDim.x*blockDim.y){
            int jid6 = 6*jid;
            fx_times_v_peq<T>(&s_vaf[84 + jid6], &s_vaf[jid6], &s_temp[jid6]);
            if (d_f_ext != nullptr) {
                for (int r = 0; r < 6; r++) { s_vaf[84 + jid6 + r] -= d_f_ext[jid6 + r]; }
            }
        }
        __syncthreads();
        //
        // Backward Pass
        //
        // s_f update where bfs_level is 6
        //     joints are: A7
        //     links are: L7
        // s_f[parent_k] += X[k]^T*f[k]
        grid_linalg_gemv<T,6,6,true>(&s_XImats[216], &s_vaf[120], &s_vaf[114], static_cast<T>(1), static_cast<T>(1));
        // s_f update where bfs_level is 5
        //     joints are: A6
        //     links are: L6
        // s_f[parent_k] += X[k]^T*f[k]
        grid_linalg_gemv<T,6,6,true>(&s_XImats[180], &s_vaf[114], &s_vaf[108], static_cast<T>(1), static_cast<T>(1));
        // s_f update where bfs_level is 4
        //     joints are: A5
        //     links are: L5
        // s_f[parent_k] += X[k]^T*f[k]
        grid_linalg_gemv<T,6,6,true>(&s_XImats[144], &s_vaf[108], &s_vaf[102], static_cast<T>(1), static_cast<T>(1));
        // s_f update where bfs_level is 3
        //     joints are: A4
        //     links are: L4
        // s_f[parent_k] += X[k]^T*f[k]
        grid_linalg_gemv<T,6,6,true>(&s_XImats[108], &s_vaf[102], &s_vaf[96], static_cast<T>(1), static_cast<T>(1));
        // s_f update where bfs_level is 2
        //     joints are: A3
        //     links are: L3
        // s_f[parent_k] += X[k]^T*f[k]
        grid_linalg_gemv<T,6,6,true>(&s_XImats[72], &s_vaf[96], &s_vaf[90], static_cast<T>(1), static_cast<T>(1));
        // s_f update where bfs_level is 1
        //     joints are: A2
        //     links are: L2
        // s_f[parent_k] += X[k]^T*f[k]
        grid_linalg_gemv<T,6,6,true>(&s_XImats[36], &s_vaf[90], &s_vaf[84], static_cast<T>(1), static_cast<T>(1));
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param s_c is the vector of output torques
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param s_qdd is the vector of joint accelerations
     * @param d_f_ext is the (optional) GLOBAL external forces, body-major 6*NUM_BODIES local-frame, or nullptr
     * @param gravity is the gravity constant
     */
    template <typename T>
    __device__
    void inverse_dynamics_device(T *s_c,  const T *s_q, const T *s_qd, const T *s_qdd, const robotModel<T> *d_robotModel, T *d_f_ext, const T gravity) {
        // GRID shared arena layout
        //   T s_vaf[126]
        //   T s_XImats[504]
        //   T s_temp[42]
        //   bytes s_linalg_smem[GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()]
        extern __shared__ __align__(16) unsigned char s_arena[];
        size_t s_arena_offset = 0;
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_vaf = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(126);
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_XImats = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(504);
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_temp = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(42);
        int *s_topology_helpers = nullptr;
        unsigned char *s_linalg_smem = nullptr;
        if (static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) > 0) {
            s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
            s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
            s_arena_offset += static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());
        }
        #ifdef GRID_CUDA_DEBUG_LAYOUT
        assert(s_arena_offset == grid_shared_arena_bytes<T>(672, 0, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()));
        #endif
        (void)s_arena_offset;
        load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
        inverse_dynamics_inner<T>(s_c, s_vaf, s_q, s_qd, s_qdd, s_XImats, s_topology_helpers, s_temp, d_f_ext, gravity);
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * Notes:
     *   optimized for qdd = 0
     *
     * @param s_c is the vector of output torques
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param d_f_ext is the (optional) GLOBAL external forces, body-major 6*NUM_BODIES local-frame, or nullptr
     * @param gravity is the gravity constant
     */
    template <typename T>
    __device__
    void inverse_dynamics_device(T *s_c,  const T *s_q, const T *s_qd, const robotModel<T> *d_robotModel, T *d_f_ext, const T gravity) {
        // GRID shared arena layout
        //   T s_vaf[126]
        //   T s_XImats[504]
        //   T s_temp[42]
        //   bytes s_linalg_smem[GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()]
        extern __shared__ __align__(16) unsigned char s_arena[];
        size_t s_arena_offset = 0;
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_vaf = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(126);
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_XImats = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(504);
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_temp = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(42);
        int *s_topology_helpers = nullptr;
        unsigned char *s_linalg_smem = nullptr;
        if (static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) > 0) {
            s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
            s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
            s_arena_offset += static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());
        }
        #ifdef GRID_CUDA_DEBUG_LAYOUT
        assert(s_arena_offset == grid_shared_arena_bytes<T>(672, 0, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()));
        #endif
        (void)s_arena_offset;
        load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
        inverse_dynamics_inner<T>(s_c, s_vaf, s_q, s_qd, s_XImats, s_topology_helpers, s_temp, d_f_ext, gravity);
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * Notes:
     *   used to compute vaf as helper values
     *
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param s_qdd is the vector of joint accelerations
     * @param d_f_ext is the (optional) GLOBAL external forces, body-major 6*NUM_BODIES local-frame, or nullptr
     * @param gravity is the gravity constant
     */
    template <typename T>
    __device__
    void inverse_dynamics_vaf_device(T *s_vaf, const T *s_q, const T *s_qd, const T *s_qdd, const robotModel<T> *d_robotModel, T *d_f_ext, const T gravity) {
        // GRID shared arena layout
        //   T s_XImats[504]
        //   T s_temp[42]
        //   bytes s_linalg_smem[GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()]
        extern __shared__ __align__(16) unsigned char s_arena[];
        size_t s_arena_offset = 0;
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_XImats = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(504);
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_temp = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(42);
        int *s_topology_helpers = nullptr;
        unsigned char *s_linalg_smem = nullptr;
        if (static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) > 0) {
            s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
            s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
            s_arena_offset += static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());
        }
        #ifdef GRID_CUDA_DEBUG_LAYOUT
        assert(s_arena_offset == grid_shared_arena_bytes<T>(546, 0, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()));
        #endif
        (void)s_arena_offset;
        load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
        inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_qdd, s_XImats, s_topology_helpers, s_temp, d_f_ext, gravity);
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * Notes:
     *   used to compute vaf as helper values
     *   optimized for qdd = 0
     *
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param d_f_ext is the (optional) GLOBAL external forces, body-major 6*NUM_BODIES local-frame, or nullptr
     * @param gravity is the gravity constant
     */
    template <typename T>
    __device__
    void inverse_dynamics_vaf_device(T *s_vaf, const T *s_q, const T *s_qd, const robotModel<T> *d_robotModel, T *d_f_ext, const T gravity) {
        // GRID shared arena layout
        //   T s_XImats[504]
        //   T s_temp[42]
        //   bytes s_linalg_smem[GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()]
        extern __shared__ __align__(16) unsigned char s_arena[];
        size_t s_arena_offset = 0;
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_XImats = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(504);
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_temp = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(42);
        int *s_topology_helpers = nullptr;
        unsigned char *s_linalg_smem = nullptr;
        if (static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) > 0) {
            s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
            s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
            s_arena_offset += static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());
        }
        #ifdef GRID_CUDA_DEBUG_LAYOUT
        assert(s_arena_offset == grid_shared_arena_bytes<T>(546, 0, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()));
        #endif
        (void)s_arena_offset;
        load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
        inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_XImats, s_topology_helpers, s_temp, d_f_ext, gravity);
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param d_c is the vector of output torques
     * @param d_q_dq is the vector of joint positions and velocities
     * @param stride_q_qd is the stide between each q, qd
     * @param d_qdd is the vector of joint accelerations
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param d_f_ext is the (optional) GLOBAL external forces, body-major 6*NUM_BODIES local-frame, or nullptr
     * @param gravity is the gravity constant,num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER, bool MUJOCO_OUTPUT = false>
    __global__
    __launch_bounds__(tier_max_threads<RESOURCE_TIER>())
    void inverse_dynamics_kernel_single_timing(T *d_c, const T *d_q_qd, const int stride_q_qd, const T *d_qdd, T *d_f_ext, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        // GRID shared arena layout
        //   T s_q_qd[14]
        //   T s_c[7]
        //   T s_vaf[126]
        //   T s_qdd[7]
        //   T s_XImats[504]
        //   T s_temp[42]
        //   bytes s_linalg_smem[GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()]
        extern __shared__ __align__(16) unsigned char s_arena[];
        size_t s_arena_offset = 0;
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_q_qd = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(14);
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_c = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(7);
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_vaf = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(126);
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_qdd = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(7);
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_XImats = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(504);
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_temp = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(42);
        int *s_topology_helpers = nullptr;
        unsigned char *s_linalg_smem = nullptr;
        if (static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) > 0) {
            s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
            s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
            s_arena_offset += static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());
        }
        #ifdef GRID_CUDA_DEBUG_LAYOUT
        assert(s_arena_offset == grid_shared_arena_bytes<T>(700, 0, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()));
        #endif
        (void)s_arena_offset;
        T *s_q = s_q_qd; T *s_qd = &s_q_qd[7];
        // load to shared mem
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
            s_q_qd[ind] = d_q_qd[ind];
        }
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
            s_qdd[ind] = d_qdd[ind];
        }
        __syncthreads();
        // compute with NUM_TIMESTEPS as NUM_REPS for timing
        for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
            // anti-LICM: volatile reload of inputs each rep
            for(int _aopt_i = threadIdx.x + threadIdx.y*blockDim.x; _aopt_i < 14; _aopt_i += blockDim.x*blockDim.y){
                reinterpret_cast<volatile T *>(s_q_qd)[_aopt_i] = reinterpret_cast<const volatile T *>(d_q_qd)[_aopt_i];
            }
            for(int _aopt_i = threadIdx.x + threadIdx.y*blockDim.x; _aopt_i < 7; _aopt_i += blockDim.x*blockDim.y){
                reinterpret_cast<volatile T *>(s_qdd)[_aopt_i] = reinterpret_cast<const volatile T *>(d_qdd)[_aopt_i];
            }
            __syncthreads();
            // anti-LICM (1/2): stomp one input slot with `rep`
            if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
                reinterpret_cast<volatile T *>(s_q_qd)[rep % (14)] = static_cast<T>(rep);
                reinterpret_cast<volatile T *>(s_qdd)[rep % (7)] = static_cast<T>(rep);
            }
            // anti-LICM (2/2): feedback prev rep's d_c into s_q_qd (true loop-carried dep)
            if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
                T _aopt_fb1 = reinterpret_cast<const volatile T *>(d_c)[(rep + 0x3FF) & 0x3FF];
                T _aopt_fb2 = reinterpret_cast<const volatile T *>(d_c)[(rep + 0x3FE) & 0x3FF];
                T _aopt_fb3 = reinterpret_cast<const volatile T *>(d_c)[(rep + 0x3FD) & 0x3FF];
                reinterpret_cast<volatile T *>(s_q_qd)[(rep + 1) % (14)] += _aopt_fb1;
                reinterpret_cast<volatile T *>(s_q_qd)[(rep + 2) % (14)] += _aopt_fb2;
                reinterpret_cast<volatile T *>(s_q_qd)[(rep + 3) % (14)] += _aopt_fb3;
            }
            __syncthreads();
            load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
            inverse_dynamics_inner<T>(s_c, s_vaf, s_q, s_qd, s_qdd, s_XImats, s_topology_helpers, s_temp, d_f_ext, gravity);
            __syncthreads();
            if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) { reinterpret_cast<volatile T *>(d_c)[rep & 1023] = reinterpret_cast<const volatile T *>(s_c)[rep & 7]; }
        }
        // save down to global
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
            d_c[ind] = s_c[ind];
        }
        __syncthreads();
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param d_c is the vector of output torques
     * @param d_q_dq is the vector of joint positions and velocities
     * @param stride_q_qd is the stide between each q, qd
     * @param d_qdd is the vector of joint accelerations
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param d_f_ext is the (optional) GLOBAL external forces, body-major 6*NUM_BODIES local-frame, or nullptr
     * @param gravity is the gravity constant,num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER, bool MUJOCO_OUTPUT = false>
    __global__
    __launch_bounds__(tier_max_threads<RESOURCE_TIER>())
    void inverse_dynamics_kernel(T *d_c, const T *d_q_qd, const int stride_q_qd, const T *d_qdd, T *d_f_ext, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        // GRID shared arena layout
        //   T s_q_qd[14]
        //   T s_c[7]
        //   T s_vaf[126]
        //   T s_qdd[7]
        //   T s_XImats[504]
        //   T s_temp[42]
        //   bytes s_linalg_smem[GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()]
        extern __shared__ __align__(16) unsigned char s_arena[];
        size_t s_arena_offset = 0;
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_q_qd = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(14);
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_c = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(7);
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_vaf = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(126);
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_qdd = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(7);
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_XImats = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(504);
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_temp = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(42);
        int *s_topology_helpers = nullptr;
        unsigned char *s_linalg_smem = nullptr;
        if (static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) > 0) {
            s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
            s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
            s_arena_offset += static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());
        }
        #ifdef GRID_CUDA_DEBUG_LAYOUT
        assert(s_arena_offset == grid_shared_arena_bytes<T>(700, 0, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()));
        #endif
        (void)s_arena_offset;
        T *s_q = s_q_qd; T *s_qd = &s_q_qd[7];
        for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
            // load to shared mem
            const T *d_q_qd_k = &d_q_qd[k*stride_q_qd];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
                s_q_qd[ind] = d_q_qd_k[ind];
            }
            const T *d_qdd_k = &d_qdd[k*7];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                s_qdd[ind] = d_qdd_k[ind];
            }
            __syncthreads();
            // compute
            load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
            inverse_dynamics_inner<T>(s_c, s_vaf, s_q, s_qd, s_qdd, s_XImats, s_topology_helpers, s_temp, d_f_ext, gravity);
            __syncthreads();
            // save down to global
            T *d_c_k = &d_c[k*7];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                d_c_k[ind] = s_c[ind];
            }
            __syncthreads();
        }
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * Notes:
     *   optimized for qdd = 0
     *
     * @param d_c is the vector of output torques
     * @param d_q_dq is the vector of joint positions and velocities
     * @param stride_q_qd is the stide between each q, qd
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param d_f_ext is the (optional) GLOBAL external forces, body-major 6*NUM_BODIES local-frame, or nullptr
     * @param gravity is the gravity constant,num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER, bool MUJOCO_OUTPUT = false>
    __global__
    __launch_bounds__(tier_max_threads<RESOURCE_TIER>())
    void inverse_dynamics_kernel_single_timing(T *d_c, const T *d_q_qd, const int stride_q_qd, T *d_f_ext, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        // GRID shared arena layout
        //   T s_q_qd[14]
        //   T s_c[7]
        //   T s_vaf[126]
        //   T s_XImats[504]
        //   T s_temp[42]
        //   bytes s_linalg_smem[GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()]
        extern __shared__ __align__(16) unsigned char s_arena[];
        size_t s_arena_offset = 0;
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_q_qd = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(14);
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_c = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(7);
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_vaf = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(126);
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_XImats = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(504);
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_temp = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(42);
        int *s_topology_helpers = nullptr;
        unsigned char *s_linalg_smem = nullptr;
        if (static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) > 0) {
            s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
            s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
            s_arena_offset += static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());
        }
        #ifdef GRID_CUDA_DEBUG_LAYOUT
        assert(s_arena_offset == grid_shared_arena_bytes<T>(693, 0, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()));
        #endif
        (void)s_arena_offset;
        T *s_q = s_q_qd; T *s_qd = &s_q_qd[7];
        // load to shared mem
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
            s_q_qd[ind] = d_q_qd[ind];
        }
        __syncthreads();
        // compute with NUM_TIMESTEPS as NUM_REPS for timing
        for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
            // anti-LICM: volatile reload of inputs each rep
            for(int _aopt_i = threadIdx.x + threadIdx.y*blockDim.x; _aopt_i < 14; _aopt_i += blockDim.x*blockDim.y){
                reinterpret_cast<volatile T *>(s_q_qd)[_aopt_i] = reinterpret_cast<const volatile T *>(d_q_qd)[_aopt_i];
            }
            __syncthreads();
            // anti-LICM (1/2): stomp one input slot with `rep`
            if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
                reinterpret_cast<volatile T *>(s_q_qd)[rep % (14)] = static_cast<T>(rep);
            }
            // anti-LICM (2/2): feedback prev rep's d_c into s_q_qd (true loop-carried dep)
            if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
                T _aopt_fb1 = reinterpret_cast<const volatile T *>(d_c)[(rep + 0x3FF) & 0x3FF];
                T _aopt_fb2 = reinterpret_cast<const volatile T *>(d_c)[(rep + 0x3FE) & 0x3FF];
                T _aopt_fb3 = reinterpret_cast<const volatile T *>(d_c)[(rep + 0x3FD) & 0x3FF];
                reinterpret_cast<volatile T *>(s_q_qd)[(rep + 1) % (14)] += _aopt_fb1;
                reinterpret_cast<volatile T *>(s_q_qd)[(rep + 2) % (14)] += _aopt_fb2;
                reinterpret_cast<volatile T *>(s_q_qd)[(rep + 3) % (14)] += _aopt_fb3;
            }
            __syncthreads();
            load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
            inverse_dynamics_inner<T>(s_c, s_vaf, s_q, s_qd, s_XImats, s_topology_helpers, s_temp, d_f_ext, gravity);
            __syncthreads();
            if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) { reinterpret_cast<volatile T *>(d_c)[rep & 1023] = reinterpret_cast<const volatile T *>(s_c)[rep & 7]; }
        }
        // save down to global
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
            d_c[ind] = s_c[ind];
        }
        __syncthreads();
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * Notes:
     *   optimized for qdd = 0
     *
     * @param d_c is the vector of output torques
     * @param d_q_dq is the vector of joint positions and velocities
     * @param stride_q_qd is the stide between each q, qd
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param d_f_ext is the (optional) GLOBAL external forces, body-major 6*NUM_BODIES local-frame, or nullptr
     * @param gravity is the gravity constant,num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER, bool MUJOCO_OUTPUT = false>
    __global__
    __launch_bounds__(tier_max_threads<RESOURCE_TIER>())
    void inverse_dynamics_kernel(T *d_c, const T *d_q_qd, const int stride_q_qd, T *d_f_ext, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        // GRID shared arena layout
        //   T s_q_qd[14]
        //   T s_c[7]
        //   T s_vaf[126]
        //   T s_XImats[504]
        //   T s_temp[42]
        //   bytes s_linalg_smem[GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()]
        extern __shared__ __align__(16) unsigned char s_arena[];
        size_t s_arena_offset = 0;
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_q_qd = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(14);
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_c = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(7);
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_vaf = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(126);
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_XImats = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(504);
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_temp = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(42);
        int *s_topology_helpers = nullptr;
        unsigned char *s_linalg_smem = nullptr;
        if (static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) > 0) {
            s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
            s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
            s_arena_offset += static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());
        }
        #ifdef GRID_CUDA_DEBUG_LAYOUT
        assert(s_arena_offset == grid_shared_arena_bytes<T>(693, 0, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()));
        #endif
        (void)s_arena_offset;
        T *s_q = s_q_qd; T *s_qd = &s_q_qd[7];
        for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
            // load to shared mem
            const T *d_q_qd_k = &d_q_qd[k*stride_q_qd];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
                s_q_qd[ind] = d_q_qd_k[ind];
            }
            __syncthreads();
            // compute
            load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
            inverse_dynamics_inner<T>(s_c, s_vaf, s_q, s_qd, s_XImats, s_topology_helpers, s_temp, d_f_ext, gravity);
            __syncthreads();
            // save down to global
            T *d_c_k = &d_c[k*7];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                d_c_k[ind] = s_c[ind];
            }
            __syncthreads();
        }
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_QDD_FLAG = false, bool USE_COMPRESSED_MEM = false, gridDataKind KIND = GRID_DATA_ALL, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>
    __host__
    void inverse_dynamics(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                          const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_DYNAMICS, "inverse_dynamics requires all-data or dynamics gridData");
        // start code with memory transfer
        int stride_q_qd;
        if (USE_COMPRESSED_MEM) {stride_q_qd = 2*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd,hd_data->h_q_qd,stride_q_qd*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        else {stride_q_qd = 3*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        if (USE_QDD_FLAG) {gpuErrchk(cudaMemcpyAsync(hd_data->d_qdd,hd_data->h_qdd,NUM_JOINTS*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[1]));}
        gpuErrchkKernel();
        // then call the kernel
        gpuErrchk(grid_check_dynamic_shared_memory_bytes("inverse_dynamics", INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>()));
        if (USE_QDD_FLAG) {
            if (USE_COMPRESSED_MEM) {inverse_dynamics_kernel<T, RESOURCE_TIER><<<block_dimms,thread_dimms,INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>()>>>(hd_data->d_c,hd_data->d_q_qd,stride_q_qd,hd_data->d_qdd, hd_data->d_f_ext,d_robotModel,gravity,num_timesteps);}
            else                    {inverse_dynamics_kernel<T, RESOURCE_TIER><<<block_dimms,thread_dimms,INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>()>>>(hd_data->d_c,hd_data->d_q_qd_u,stride_q_qd,hd_data->d_qdd, hd_data->d_f_ext,d_robotModel,gravity,num_timesteps);}
        }
        else {
            if (USE_COMPRESSED_MEM) {inverse_dynamics_kernel<T, RESOURCE_TIER><<<block_dimms,thread_dimms,INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>()>>>(hd_data->d_c,hd_data->d_q_qd,stride_q_qd,hd_data->d_f_ext,d_robotModel,gravity,num_timesteps);}
            else                    {inverse_dynamics_kernel<T, RESOURCE_TIER><<<block_dimms,thread_dimms,INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>()>>>(hd_data->d_c,hd_data->d_q_qd_u,stride_q_qd,hd_data->d_f_ext,d_robotModel,gravity,num_timesteps);}
        }
        gpuErrchkKernel();
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_c,hd_data->d_c,NUM_JOINTS*num_timesteps*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchkKernel();
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_QDD_FLAG = false, bool USE_COMPRESSED_MEM = false, gridDataKind KIND = GRID_DATA_ALL, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>
    __host__
    void inverse_dynamics_single_timing(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                                        const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_DYNAMICS, "inverse_dynamics requires all-data or dynamics gridData");
        // start code with memory transfer
        int stride_q_qd;
        if (USE_COMPRESSED_MEM) {stride_q_qd = 2*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd,hd_data->h_q_qd,stride_q_qd*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        else {stride_q_qd = 3*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        if (USE_QDD_FLAG) {gpuErrchk(cudaMemcpyAsync(hd_data->d_qdd,hd_data->h_qdd,NUM_JOINTS*sizeof(T),cudaMemcpyHostToDevice,streams[1]));}
        gpuErrchkKernel();
        // then call the kernel
        gpuErrchk(grid_check_dynamic_shared_memory_bytes("inverse_dynamics", INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>()));
        struct timespec start, end; clock_gettime(CLOCK_MONOTONIC,&start);
        if (USE_QDD_FLAG) {
            if (USE_COMPRESSED_MEM) {inverse_dynamics_kernel_single_timing<T, RESOURCE_TIER><<<block_dimms,thread_dimms,INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>()>>>(hd_data->d_c,hd_data->d_q_qd,stride_q_qd,hd_data->d_qdd, hd_data->d_f_ext,d_robotModel,gravity,num_timesteps);}
            else                    {inverse_dynamics_kernel_single_timing<T, RESOURCE_TIER><<<block_dimms,thread_dimms,INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>()>>>(hd_data->d_c,hd_data->d_q_qd_u,stride_q_qd,hd_data->d_qdd, hd_data->d_f_ext,d_robotModel,gravity,num_timesteps);}
        }
        else {
            if (USE_COMPRESSED_MEM) {inverse_dynamics_kernel_single_timing<T, RESOURCE_TIER><<<block_dimms,thread_dimms,INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>()>>>(hd_data->d_c,hd_data->d_q_qd,stride_q_qd,hd_data->d_f_ext,d_robotModel,gravity,num_timesteps);}
            else                    {inverse_dynamics_kernel_single_timing<T, RESOURCE_TIER><<<block_dimms,thread_dimms,INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>()>>>(hd_data->d_c,hd_data->d_q_qd_u,stride_q_qd,hd_data->d_f_ext,d_robotModel,gravity,num_timesteps);}
        }
        gpuErrchkKernel();
        clock_gettime(CLOCK_MONOTONIC,&end);
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_c,hd_data->d_c,NUM_JOINTS*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchkKernel();
        printf("Single Call INVERSE_DYNAMICS %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(num_timesteps));
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_QDD_FLAG = false, bool USE_COMPRESSED_MEM = false, gridDataKind KIND = GRID_DATA_ALL, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>
    __host__
    void inverse_dynamics_compute_only(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                                       const dim3 block_dimms, const dim3 thread_dimms) {
        static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_DYNAMICS, "inverse_dynamics requires all-data or dynamics gridData");
        int stride_q_qd = USE_COMPRESSED_MEM ? 2*NUM_JOINTS: 3*NUM_JOINTS;
        // then call the kernel
        gpuErrchk(grid_check_dynamic_shared_memory_bytes("inverse_dynamics", INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>()));
        if (USE_QDD_FLAG) {
            if (USE_COMPRESSED_MEM) {inverse_dynamics_kernel<T, RESOURCE_TIER><<<block_dimms,thread_dimms,INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>()>>>(hd_data->d_c,hd_data->d_q_qd,stride_q_qd,hd_data->d_qdd, hd_data->d_f_ext,d_robotModel,gravity,num_timesteps);}
            else                    {inverse_dynamics_kernel<T, RESOURCE_TIER><<<block_dimms,thread_dimms,INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>()>>>(hd_data->d_c,hd_data->d_q_qd_u,stride_q_qd,hd_data->d_qdd, hd_data->d_f_ext,d_robotModel,gravity,num_timesteps);}
        }
        else {
            if (USE_COMPRESSED_MEM) {inverse_dynamics_kernel<T, RESOURCE_TIER><<<block_dimms,thread_dimms,INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>()>>>(hd_data->d_c,hd_data->d_q_qd,stride_q_qd,hd_data->d_f_ext,d_robotModel,gravity,num_timesteps);}
            else                    {inverse_dynamics_kernel<T, RESOURCE_TIER><<<block_dimms,thread_dimms,INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>()>>>(hd_data->d_c,hd_data->d_q_qd_u,stride_q_qd,hd_data->d_f_ext,d_robotModel,gravity,num_timesteps);}
        }
        gpuErrchkKernel();
    }

    /**
     * Compute the inverse of the mass matrix
     *
     * Notes:
     *   CALLER CONTRACT (direct *_inner callers): s_XImats must ALREADY be populated for the current s_q (READ-only here) via load_update_XImats_helpers(...) + __syncthreads(), and s_temp MUST be MINV_INNER_SMEM_BYTES<T, F_IN_SMEM>() bytes -- at F_IN_SMEM=true the 6*NV*NV F-band lives in the TAIL of s_temp (d_workspace=0/nullptr does NOT mean the band is free). Under-populating XImats or under-sizing s_temp reads never-written shared -> NaN (race-clean, DoF-specific since the band scales as 6*NV*NV). Prefer minv_device, which handles both.
     *   Outputs a SYMMETRIC_UPPER triangular matrix for Minv
     *   Inner-controlled placement: F_IN_SMEM selects where the 6*NV*NV F-region lives.
     *     true  -> tail of s_temp (shared; fastest, default).  false -> d_workspace (global).
     *   The choice is made at the top of this fn so the caller just sizes both arenas from
     *   the MINV_INNER_*_BYTES constants and hands both pointers in. Codegen maps each
     *   RESOURCE_TIER to an F_IN_SMEM value per robot (MINV_F_IN_SMEM<TIER>()).
     *
     * @param s_Minv is a pointer to memory for the final result
     * @param s_q is the vector of joint positions
     * @param s_temp is the (shared) scratch; size MINV_INNER_SMEM_BYTES<T, F_IN_SMEM>() (= 373 always, plus the 294-float F-region when F_IN_SMEM)
     * @param s_XImats is the (shared) memory holding the updated XI matricies for the given s_q
     * @param s_topology_helpers is the (shared) memory location for the topology_helpers (nullptr/unused for serial chains with identical Ss)
     * @param d_workspace is the global scratch; size MINV_INNER_WORKSPACE_BYTES<T, F_IN_SMEM>() (= 294 when !F_IN_SMEM, else 0). Pass nullptr when F_IN_SMEM
     */
    template <typename T, bool F_IN_SMEM = true>
    __device__ __forceinline__
    void minv_inner(T *s_Minv, const T *s_q, T *s_XImats, int *s_topology_helpers, T *s_temp, T *d_workspace) {
        unsigned char *s_linalg_smem = nullptr;
        T *s_F;
        if constexpr (F_IN_SMEM) { s_F = &s_temp[373]; (void)d_workspace; }
        else { s_F = d_workspace; }
        // T *s_F (param) [size 6*NV*NV]; T *s_IA = &s_temp[0]; T *s_U = &s_temp[252]; T *s_Dinv = &s_temp[294]; T *s_Ia = &s_temp[301]; T *s_IaTemp = &s_temp[337];
        // Initialize IA = I
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 252; ind += blockDim.x*blockDim.y){
            s_temp[0 + ind] = s_XImats[252 + ind];
        }
        // Zero Minv and F
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 343; ind += blockDim.x*blockDim.y){
            if(ind < 294){s_F[0 + ind] = static_cast<T>(0);}
            else{s_Minv[ind - 294] = static_cast<T>(0);}
        }
        __syncthreads();
        //
        // Backward Pass
        //
        // backward pass updates where bfs_level is 6
        //     joints are: A7
        //     links are: L7
        // U = IA*S, D = S^T*U, DInv = 1/D, Minv[i,i] = Dinv
        for(int row = threadIdx.x + threadIdx.y*blockDim.x; row < 6; row += blockDim.x*blockDim.y){
            s_temp[252 + 36 + row] = (1) * s_temp[0 + 6*36 + 6*2 + row];
            if(row == 2){
                s_temp[294 + 6] = static_cast<T>(1)/((1) * s_temp[252 + 36 + 2]);
                s_Minv[8 * 6] = s_temp[294 + 6];
            }
        }
        __syncthreads();
        // Minv[i,subTreeInds] -= Dinv*F[i,Srow,SubTreeInds]
        // Temp Comp: F[i,:,subTreeInds] += U*Minv[i,subTreeInds] - to start Fparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            s_Minv[42 + 6] -= s_temp[294 + 6] * (1) * s_F[0 + 42*6 + 36 + 2];
            for(int row = 0; row < 6; row++) {
                s_F[0 + 42*6 + 36 + row] += s_temp[252 + 6*6 + row] * s_Minv[42 + 6];
            }
        }
        // Ia = IA - U^T Dinv U | to start IAparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6;
            s_temp[301 + ind] = s_temp[216 + ind] - (s_temp[288 + row] * s_temp[300] * s_temp[288 + col]);
        }
        __syncthreads();
        // F[parent_ind,:,subTreeInds] += Xmat^T * F[ind,:,subTreeInds]
        // IA_Update_Temp = Xmat^T * Ia | for IAparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 42; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6;
            T *src = &s_F[0 + 42*6 + 6*6]; T *dst = &s_F[0 + 42*5 + 6*6];
            // adjust for temp comps
            if (col >= 1) {
                col -= 1; src = &s_temp[301 + 6*col]; dst = &s_temp[337 + 6*col];
            }
            dst[row] = dot_prod<T,6,1,1>(&s_XImats[36*6 + 6*row],src);
        }
        __syncthreads();
        // IA[parent_ind] += IA_Update_Temp * Xmat
        grid_linalg_gemm<T,6,6,6>(&s_temp[337], &s_XImats[216], &s_temp[180], static_cast<T>(1), static_cast<T>(1), s_linalg_smem);
        // backward pass updates where bfs_level is 5
        //     joints are: A6
        //     links are: L6
        // U = IA*S, D = S^T*U, DInv = 1/D, Minv[i,i] = Dinv
        for(int row = threadIdx.x + threadIdx.y*blockDim.x; row < 6; row += blockDim.x*blockDim.y){
            s_temp[252 + 30 + row] = (1) * s_temp[0 + 6*30 + 6*2 + row];
            if(row == 2){
                s_temp[294 + 5] = static_cast<T>(1)/((1) * s_temp[252 + 30 + 2]);
                s_Minv[8 * 5] = s_temp[294 + 5];
            }
        }
        __syncthreads();
        // Minv[i,subTreeInds] -= Dinv*F[i,Srow,SubTreeInds]
        // Temp Comp: F[i,:,subTreeInds] += U*Minv[i,subTreeInds] - to start Fparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 2; ind += blockDim.x*blockDim.y){
            int jid_subtree6 = 6*(5 + ind); int jid_subtreeN = 7*(5 + ind);
            s_Minv[jid_subtreeN + 5] -= s_temp[294 + 5] * (1) * s_F[0 + 42*5 + jid_subtree6 + 2];
            for(int row = 0; row < 6; row++) {
                s_F[0 + 42*5 + jid_subtree6 + row] += s_temp[252 + 6*5 + row] * s_Minv[jid_subtreeN + 5];
            }
        }
        // Ia = IA - U^T Dinv U | to start IAparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6;
            s_temp[301 + ind] = s_temp[180 + ind] - (s_temp[282 + row] * s_temp[299] * s_temp[282 + col]);
        }
        __syncthreads();
        // F[parent_ind,:,subTreeInds] += Xmat^T * F[ind,:,subTreeInds]
        // IA_Update_Temp = Xmat^T * Ia | for IAparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 48; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6;
            T *src = &s_F[0 + 42*5 + 6*(5 + col)]; T *dst = &s_F[0 + 42*4 + 6*(5 + col)];
            // adjust for temp comps
            if (col >= 2) {
                col -= 2; src = &s_temp[301 + 6*col]; dst = &s_temp[337 + 6*col];
            }
            dst[row] = dot_prod<T,6,1,1>(&s_XImats[36*5 + 6*row],src);
        }
        __syncthreads();
        // IA[parent_ind] += IA_Update_Temp * Xmat
        grid_linalg_gemm<T,6,6,6>(&s_temp[337], &s_XImats[180], &s_temp[144], static_cast<T>(1), static_cast<T>(1), s_linalg_smem);
        // backward pass updates where bfs_level is 4
        //     joints are: A5
        //     links are: L5
        // U = IA*S, D = S^T*U, DInv = 1/D, Minv[i,i] = Dinv
        for(int row = threadIdx.x + threadIdx.y*blockDim.x; row < 6; row += blockDim.x*blockDim.y){
            s_temp[252 + 24 + row] = (1) * s_temp[0 + 6*24 + 6*2 + row];
            if(row == 2){
                s_temp[294 + 4] = static_cast<T>(1)/((1) * s_temp[252 + 24 + 2]);
                s_Minv[8 * 4] = s_temp[294 + 4];
            }
        }
        __syncthreads();
        // Minv[i,subTreeInds] -= Dinv*F[i,Srow,SubTreeInds]
        // Temp Comp: F[i,:,subTreeInds] += U*Minv[i,subTreeInds] - to start Fparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 3; ind += blockDim.x*blockDim.y){
            int jid_subtree6 = 6*(4 + ind); int jid_subtreeN = 7*(4 + ind);
            s_Minv[jid_subtreeN + 4] -= s_temp[294 + 4] * (1) * s_F[0 + 42*4 + jid_subtree6 + 2];
            for(int row = 0; row < 6; row++) {
                s_F[0 + 42*4 + jid_subtree6 + row] += s_temp[252 + 6*4 + row] * s_Minv[jid_subtreeN + 4];
            }
        }
        // Ia = IA - U^T Dinv U | to start IAparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6;
            s_temp[301 + ind] = s_temp[144 + ind] - (s_temp[276 + row] * s_temp[298] * s_temp[276 + col]);
        }
        __syncthreads();
        // F[parent_ind,:,subTreeInds] += Xmat^T * F[ind,:,subTreeInds]
        // IA_Update_Temp = Xmat^T * Ia | for IAparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 54; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6;
            T *src = &s_F[0 + 42*4 + 6*(4 + col)]; T *dst = &s_F[0 + 42*3 + 6*(4 + col)];
            // adjust for temp comps
            if (col >= 3) {
                col -= 3; src = &s_temp[301 + 6*col]; dst = &s_temp[337 + 6*col];
            }
            dst[row] = dot_prod<T,6,1,1>(&s_XImats[36*4 + 6*row],src);
        }
        __syncthreads();
        // IA[parent_ind] += IA_Update_Temp * Xmat
        grid_linalg_gemm<T,6,6,6>(&s_temp[337], &s_XImats[144], &s_temp[108], static_cast<T>(1), static_cast<T>(1), s_linalg_smem);
        // backward pass updates where bfs_level is 3
        //     joints are: A4
        //     links are: L4
        // U = IA*S, D = S^T*U, DInv = 1/D, Minv[i,i] = Dinv
        for(int row = threadIdx.x + threadIdx.y*blockDim.x; row < 6; row += blockDim.x*blockDim.y){
            s_temp[252 + 18 + row] = (1) * s_temp[0 + 6*18 + 6*2 + row];
            if(row == 2){
                s_temp[294 + 3] = static_cast<T>(1)/((1) * s_temp[252 + 18 + 2]);
                s_Minv[8 * 3] = s_temp[294 + 3];
            }
        }
        __syncthreads();
        // Minv[i,subTreeInds] -= Dinv*F[i,Srow,SubTreeInds]
        // Temp Comp: F[i,:,subTreeInds] += U*Minv[i,subTreeInds] - to start Fparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 4; ind += blockDim.x*blockDim.y){
            int jid_subtree6 = 6*(3 + ind); int jid_subtreeN = 7*(3 + ind);
            s_Minv[jid_subtreeN + 3] -= s_temp[294 + 3] * (1) * s_F[0 + 42*3 + jid_subtree6 + 2];
            for(int row = 0; row < 6; row++) {
                s_F[0 + 42*3 + jid_subtree6 + row] += s_temp[252 + 6*3 + row] * s_Minv[jid_subtreeN + 3];
            }
        }
        // Ia = IA - U^T Dinv U | to start IAparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6;
            s_temp[301 + ind] = s_temp[108 + ind] - (s_temp[270 + row] * s_temp[297] * s_temp[270 + col]);
        }
        __syncthreads();
        // F[parent_ind,:,subTreeInds] += Xmat^T * F[ind,:,subTreeInds]
        // IA_Update_Temp = Xmat^T * Ia | for IAparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 60; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6;
            T *src = &s_F[0 + 42*3 + 6*(3 + col)]; T *dst = &s_F[0 + 42*2 + 6*(3 + col)];
            // adjust for temp comps
            if (col >= 4) {
                col -= 4; src = &s_temp[301 + 6*col]; dst = &s_temp[337 + 6*col];
            }
            dst[row] = dot_prod<T,6,1,1>(&s_XImats[36*3 + 6*row],src);
        }
        __syncthreads();
        // IA[parent_ind] += IA_Update_Temp * Xmat
        grid_linalg_gemm<T,6,6,6>(&s_temp[337], &s_XImats[108], &s_temp[72], static_cast<T>(1), static_cast<T>(1), s_linalg_smem);
        // backward pass updates where bfs_level is 2
        //     joints are: A3
        //     links are: L3
        // U = IA*S, D = S^T*U, DInv = 1/D, Minv[i,i] = Dinv
        for(int row = threadIdx.x + threadIdx.y*blockDim.x; row < 6; row += blockDim.x*blockDim.y){
            s_temp[252 + 12 + row] = (1) * s_temp[0 + 6*12 + 6*2 + row];
            if(row == 2){
                s_temp[294 + 2] = static_cast<T>(1)/((1) * s_temp[252 + 12 + 2]);
                s_Minv[8 * 2] = s_temp[294 + 2];
            }
        }
        __syncthreads();
        // Minv[i,subTreeInds] -= Dinv*F[i,Srow,SubTreeInds]
        // Temp Comp: F[i,:,subTreeInds] += U*Minv[i,subTreeInds] - to start Fparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 5; ind += blockDim.x*blockDim.y){
            int jid_subtree6 = 6*(2 + ind); int jid_subtreeN = 7*(2 + ind);
            s_Minv[jid_subtreeN + 2] -= s_temp[294 + 2] * (1) * s_F[0 + 42*2 + jid_subtree6 + 2];
            for(int row = 0; row < 6; row++) {
                s_F[0 + 42*2 + jid_subtree6 + row] += s_temp[252 + 6*2 + row] * s_Minv[jid_subtreeN + 2];
            }
        }
        // Ia = IA - U^T Dinv U | to start IAparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6;
            s_temp[301 + ind] = s_temp[72 + ind] - (s_temp[264 + row] * s_temp[296] * s_temp[264 + col]);
        }
        __syncthreads();
        // F[parent_ind,:,subTreeInds] += Xmat^T * F[ind,:,subTreeInds]
        // IA_Update_Temp = Xmat^T * Ia | for IAparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 66; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6;
            T *src = &s_F[0 + 42*2 + 6*(2 + col)]; T *dst = &s_F[0 + 42*1 + 6*(2 + col)];
            // adjust for temp comps
            if (col >= 5) {
                col -= 5; src = &s_temp[301 + 6*col]; dst = &s_temp[337 + 6*col];
            }
            dst[row] = dot_prod<T,6,1,1>(&s_XImats[36*2 + 6*row],src);
        }
        __syncthreads();
        // IA[parent_ind] += IA_Update_Temp * Xmat
        grid_linalg_gemm<T,6,6,6>(&s_temp[337], &s_XImats[72], &s_temp[36], static_cast<T>(1), static_cast<T>(1), s_linalg_smem);
        // backward pass updates where bfs_level is 1
        //     joints are: A2
        //     links are: L2
        // U = IA*S, D = S^T*U, DInv = 1/D, Minv[i,i] = Dinv
        for(int row = threadIdx.x + threadIdx.y*blockDim.x; row < 6; row += blockDim.x*blockDim.y){
            s_temp[252 + 6 + row] = (1) * s_temp[0 + 6*6 + 6*2 + row];
            if(row == 2){
                s_temp[294 + 1] = static_cast<T>(1)/((1) * s_temp[252 + 6 + 2]);
                s_Minv[8 * 1] = s_temp[294 + 1];
            }
        }
        __syncthreads();
        // Minv[i,subTreeInds] -= Dinv*F[i,Srow,SubTreeInds]
        // Temp Comp: F[i,:,subTreeInds] += U*Minv[i,subTreeInds] - to start Fparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int jid_subtree6 = 6*(1 + ind); int jid_subtreeN = 7*(1 + ind);
            s_Minv[jid_subtreeN + 1] -= s_temp[294 + 1] * (1) * s_F[0 + 42*1 + jid_subtree6 + 2];
            for(int row = 0; row < 6; row++) {
                s_F[0 + 42*1 + jid_subtree6 + row] += s_temp[252 + 6*1 + row] * s_Minv[jid_subtreeN + 1];
            }
        }
        // Ia = IA - U^T Dinv U | to start IAparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6;
            s_temp[301 + ind] = s_temp[36 + ind] - (s_temp[258 + row] * s_temp[295] * s_temp[258 + col]);
        }
        __syncthreads();
        // F[parent_ind,:,subTreeInds] += Xmat^T * F[ind,:,subTreeInds]
        // IA_Update_Temp = Xmat^T * Ia | for IAparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 72; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6;
            T *src = &s_F[0 + 42*1 + 6*(1 + col)]; T *dst = &s_F[0 + 42*0 + 6*(1 + col)];
            // adjust for temp comps
            if (col >= 6) {
                col -= 6; src = &s_temp[301 + 6*col]; dst = &s_temp[337 + 6*col];
            }
            dst[row] = dot_prod<T,6,1,1>(&s_XImats[36*1 + 6*row],src);
        }
        __syncthreads();
        // IA[parent_ind] += IA_Update_Temp * Xmat
        grid_linalg_gemm<T,6,6,6>(&s_temp[337], &s_XImats[36], &s_temp[0], static_cast<T>(1), static_cast<T>(1), s_linalg_smem);
        // backward pass updates where bfs_level is 0
        //     joints are: A1
        //     links are: L1
        // U = IA*S, D = S^T*U, DInv = 1/D, Minv[i,i] = Dinv
        for(int row = threadIdx.x + threadIdx.y*blockDim.x; row < 6; row += blockDim.x*blockDim.y){
            s_temp[252 + 0 + row] = (1) * s_temp[0 + 6*0 + 6*2 + row];
            if(row == 2){
                s_temp[294 + 0] = static_cast<T>(1)/((1) * s_temp[252 + 0 + 2]);
                s_Minv[8 * 0] = s_temp[294 + 0];
            }
        }
        __syncthreads();
        // Minv[i,subTreeInds] -= Dinv*F[i,Srow,SubTreeInds]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
            int jid_subtree6 = 6*(0 + ind); int jid_subtreeN = 7*(0 + ind);
            s_Minv[jid_subtreeN + 0] -= s_temp[294 + 0] * (1) * s_F[0 + 42*0 + jid_subtree6 + 2];
        }
        __syncthreads();
        //
        // Forward Pass
        //   Note that due to the i: operation we need to go serially over all n
        //
        // forward pass for jid: 0
        // F[i,:,i:] = S * Minv[i,i:] as parent is base so rest is skipped
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 42; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6;
            s_F[0 + ind] = (row == 2) * (1) * s_Minv[0 + 7 * col];
        }
        __syncthreads();
        // forward pass for jid: 1
        // Minv[i,i:] -= Dinv*U^T*Xmat*F[parent,:,i:] across cols i...N
        // F[i,:,i:] = S * Minv[i,i:] + Xmat*F[parent,:,i:] across cols i...N
        //   Per column: F[i,:,col]=Xmat*F[parent,:,col], then
        //   Minv[i,col]-=Dinv*U^T*F[i,:,col] and F[i,Srow,col]+=S*Minv[i,col]
        for(int c = threadIdx.x + threadIdx.y*blockDim.x; c < 6; c += blockDim.x*blockDim.y){
            int col_ind = c + 1;
            T *s_Fcol = &s_F[42 + 6*col_ind];
            T *s_Fpcol = &s_F[0 + 6*col_ind];
            for (int row = 0; row < 6; row++) {
                s_Fcol[row] = dot_prod<T,6,6,1>(&s_XImats[36 + row], s_Fpcol);
            }
            s_Minv[7 * col_ind + 1] -= s_temp[295] * dot_prod<T,6,1,1>(s_Fcol,&s_temp[258]);
            s_Fcol[2] += (1) * s_Minv[7 * col_ind + 1];
        }
        __syncthreads();
        // forward pass for jid: 2
        // Minv[i,i:] -= Dinv*U^T*Xmat*F[parent,:,i:] across cols i...N
        // F[i,:,i:] = S * Minv[i,i:] + Xmat*F[parent,:,i:] across cols i...N
        //   Per column: F[i,:,col]=Xmat*F[parent,:,col], then
        //   Minv[i,col]-=Dinv*U^T*F[i,:,col] and F[i,Srow,col]+=S*Minv[i,col]
        for(int c = threadIdx.x + threadIdx.y*blockDim.x; c < 5; c += blockDim.x*blockDim.y){
            int col_ind = c + 2;
            T *s_Fcol = &s_F[84 + 6*col_ind];
            T *s_Fpcol = &s_F[42 + 6*col_ind];
            for (int row = 0; row < 6; row++) {
                s_Fcol[row] = dot_prod<T,6,6,1>(&s_XImats[72 + row], s_Fpcol);
            }
            s_Minv[7 * col_ind + 2] -= s_temp[296] * dot_prod<T,6,1,1>(s_Fcol,&s_temp[264]);
            s_Fcol[2] += (1) * s_Minv[7 * col_ind + 2];
        }
        __syncthreads();
        // forward pass for jid: 3
        // Minv[i,i:] -= Dinv*U^T*Xmat*F[parent,:,i:] across cols i...N
        // F[i,:,i:] = S * Minv[i,i:] + Xmat*F[parent,:,i:] across cols i...N
        //   Per column: F[i,:,col]=Xmat*F[parent,:,col], then
        //   Minv[i,col]-=Dinv*U^T*F[i,:,col] and F[i,Srow,col]+=S*Minv[i,col]
        for(int c = threadIdx.x + threadIdx.y*blockDim.x; c < 4; c += blockDim.x*blockDim.y){
            int col_ind = c + 3;
            T *s_Fcol = &s_F[126 + 6*col_ind];
            T *s_Fpcol = &s_F[84 + 6*col_ind];
            for (int row = 0; row < 6; row++) {
                s_Fcol[row] = dot_prod<T,6,6,1>(&s_XImats[108 + row], s_Fpcol);
            }
            s_Minv[7 * col_ind + 3] -= s_temp[297] * dot_prod<T,6,1,1>(s_Fcol,&s_temp[270]);
            s_Fcol[2] += (1) * s_Minv[7 * col_ind + 3];
        }
        __syncthreads();
        // forward pass for jid: 4
        // Minv[i,i:] -= Dinv*U^T*Xmat*F[parent,:,i:] across cols i...N
        // F[i,:,i:] = S * Minv[i,i:] + Xmat*F[parent,:,i:] across cols i...N
        //   Per column: F[i,:,col]=Xmat*F[parent,:,col], then
        //   Minv[i,col]-=Dinv*U^T*F[i,:,col] and F[i,Srow,col]+=S*Minv[i,col]
        for(int c = threadIdx.x + threadIdx.y*blockDim.x; c < 3; c += blockDim.x*blockDim.y){
            int col_ind = c + 4;
            T *s_Fcol = &s_F[168 + 6*col_ind];
            T *s_Fpcol = &s_F[126 + 6*col_ind];
            for (int row = 0; row < 6; row++) {
                s_Fcol[row] = dot_prod<T,6,6,1>(&s_XImats[144 + row], s_Fpcol);
            }
            s_Minv[7 * col_ind + 4] -= s_temp[298] * dot_prod<T,6,1,1>(s_Fcol,&s_temp[276]);
            s_Fcol[2] += (1) * s_Minv[7 * col_ind + 4];
        }
        __syncthreads();
        // forward pass for jid: 5
        // Minv[i,i:] -= Dinv*U^T*Xmat*F[parent,:,i:] across cols i...N
        // F[i,:,i:] = S * Minv[i,i:] + Xmat*F[parent,:,i:] across cols i...N
        //   Per column: F[i,:,col]=Xmat*F[parent,:,col], then
        //   Minv[i,col]-=Dinv*U^T*F[i,:,col] and F[i,Srow,col]+=S*Minv[i,col]
        for(int c = threadIdx.x + threadIdx.y*blockDim.x; c < 2; c += blockDim.x*blockDim.y){
            int col_ind = c + 5;
            T *s_Fcol = &s_F[210 + 6*col_ind];
            T *s_Fpcol = &s_F[168 + 6*col_ind];
            for (int row = 0; row < 6; row++) {
                s_Fcol[row] = dot_prod<T,6,6,1>(&s_XImats[180 + row], s_Fpcol);
            }
            s_Minv[7 * col_ind + 5] -= s_temp[299] * dot_prod<T,6,1,1>(s_Fcol,&s_temp[282]);
            s_Fcol[2] += (1) * s_Minv[7 * col_ind + 5];
        }
        __syncthreads();
        // forward pass for jid: 6
        // Minv[i,i:] -= Dinv*U^T*Xmat*F[parent,:,i:] across cols i...N
        // F[i,:,i:] = S * Minv[i,i:] + Xmat*F[parent,:,i:] across cols i...N
        //   Per column: F[i,:,col]=Xmat*F[parent,:,col], then
        //   Minv[i,col]-=Dinv*U^T*F[i,:,col] and F[i,Srow,col]+=S*Minv[i,col]
        for(int c = threadIdx.x + threadIdx.y*blockDim.x; c < 1; c += blockDim.x*blockDim.y){
            int col_ind = c + 6;
            T *s_Fcol = &s_F[252 + 6*col_ind];
            T *s_Fpcol = &s_F[210 + 6*col_ind];
            for (int row = 0; row < 6; row++) {
                s_Fcol[row] = dot_prod<T,6,6,1>(&s_XImats[216 + row], s_Fpcol);
            }
            s_Minv[7 * col_ind + 6] -= s_temp[300] * dot_prod<T,6,1,1>(s_Fcol,&s_temp[288]);
        }
        __syncthreads();
    }

    /**
     * Compute the inverse of the mass matrix
     *
     * Notes:
     *   Outputs a SYMMETRIC_UPPER triangular matrix for Minv
     *
     * @param s_Minv is a pointer to memory for the final result
     * @param s_q is the vector of joint positions
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     */
    template <typename T>
    __device__
    void minv_device(T *s_Minv, const T *s_q, const robotModel<T> *d_robotModel){
        // GRID shared arena layout
        //   T s_XImats[504]
        //   T s_temp[667]
        //   bytes s_linalg_smem[GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()]
        extern __shared__ __align__(16) unsigned char s_arena[];
        size_t s_arena_offset = 0;
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_XImats = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(504);
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_temp = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(667);
        int *s_topology_helpers = nullptr;
        unsigned char *s_linalg_smem = nullptr;
        if (static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) > 0) {
            s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
            s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
            s_arena_offset += static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());
        }
        #ifdef GRID_CUDA_DEBUG_LAYOUT
        assert(s_arena_offset == grid_shared_arena_bytes<T>(1171, 0, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()));
        #endif
        (void)s_arena_offset;
        load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
        minv_inner<T, true>(s_Minv, s_q, s_XImats, s_topology_helpers, s_temp, nullptr);
    }

    /**
     * Compute the inverse of the mass matrix
     *
     * Notes:
     *   Outputs a SYMMETRIC_UPPER triangular matrix for Minv
     *
     * @param d_Minv is a pointer to memory for the final result
     * @param d_workspace is the L2-pinned global spill buffer (used when Minv-F overflows smem)
     * @param d_q is the vector of joint positions
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER, bool MUJOCO_OUTPUT = false>
    __global__
    __launch_bounds__(tier_max_threads<RESOURCE_TIER>())
    void minv_kernel_single_timing(T *d_Minv, unsigned char *d_workspace, const T *d_q, const int stride_q, const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS){
        if constexpr (RESOURCE_TIER == TIER_SHARED) {
            // GRID shared arena layout
            //   T s_q[7]
            //   T s_Minv[49]
            //   T s_XImats[504]
            //   T s_temp[667]
            //   bytes s_linalg_smem[GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()]
            extern __shared__ __align__(16) unsigned char s_arena[];
            size_t s_arena_offset = 0;
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_q = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(7);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_Minv = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(49);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_XImats = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(504);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_temp = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(667);
            int *s_topology_helpers = nullptr;
            unsigned char *s_linalg_smem = nullptr;
            if (static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) > 0) {
                s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
                s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
                s_arena_offset += static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());
            }
            #ifdef GRID_CUDA_DEBUG_LAYOUT
            assert(s_arena_offset == grid_shared_arena_bytes<T>(1227, 0, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()));
            #endif
            (void)s_arena_offset;
            // load to shared mem
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                s_q[ind] = d_q[ind];
            }
            __syncthreads();
            (void)d_workspace;
            // compute with NUM_TIMESTEPS as NUM_REPS for timing
            for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
                // anti-LICM: volatile reload of inputs each rep
                for(int _aopt_i = threadIdx.x + threadIdx.y*blockDim.x; _aopt_i < 7; _aopt_i += blockDim.x*blockDim.y){
                    reinterpret_cast<volatile T *>(s_q)[_aopt_i] = reinterpret_cast<const volatile T *>(d_q)[_aopt_i];
                }
                __syncthreads();
                // anti-LICM (1/2): stomp one input slot with `rep`
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
                    reinterpret_cast<volatile T *>(s_q)[rep % (7)] = static_cast<T>(rep);
                }
                // anti-LICM (2/2): feedback prev rep's d_Minv into s_q (true loop-carried dep)
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
                    T _aopt_fb1 = reinterpret_cast<const volatile T *>(d_Minv)[(rep + 0x3FF) & 0x3FF];
                    T _aopt_fb2 = reinterpret_cast<const volatile T *>(d_Minv)[(rep + 0x3FE) & 0x3FF];
                    T _aopt_fb3 = reinterpret_cast<const volatile T *>(d_Minv)[(rep + 0x3FD) & 0x3FF];
                    reinterpret_cast<volatile T *>(s_q)[(rep + 1) % (7)] += _aopt_fb1;
                    reinterpret_cast<volatile T *>(s_q)[(rep + 2) % (7)] += _aopt_fb2;
                    reinterpret_cast<volatile T *>(s_q)[(rep + 3) % (7)] += _aopt_fb3;
                }
                __syncthreads();
                load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
                minv_inner<T, true>(s_Minv, s_q, s_XImats, s_topology_helpers, s_temp, nullptr);
                __syncthreads();
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) { reinterpret_cast<volatile T *>(d_Minv)[rep & 1023] = reinterpret_cast<const volatile T *>(s_Minv)[rep & 7]; }
            }
            // save down to global
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 49; ind += blockDim.x*blockDim.y){
                d_Minv[ind] = s_Minv[ind];
            }
            __syncthreads();
        }
        else if constexpr (RESOURCE_TIER == TIER_LITE) {
            // GRID shared arena layout
            //   T s_q[7]
            //   T s_Minv[49]
            //   T s_XImats[504]
            //   T s_temp[667]
            //   bytes s_linalg_smem[GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()]
            extern __shared__ __align__(16) unsigned char s_arena[];
            size_t s_arena_offset = 0;
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_q = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(7);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_Minv = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(49);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_XImats = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(504);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_temp = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(667);
            int *s_topology_helpers = nullptr;
            unsigned char *s_linalg_smem = nullptr;
            if (static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) > 0) {
                s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
                s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
                s_arena_offset += static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());
            }
            #ifdef GRID_CUDA_DEBUG_LAYOUT
            assert(s_arena_offset == grid_shared_arena_bytes<T>(1227, 0, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()));
            #endif
            (void)s_arena_offset;
            // load to shared mem
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                s_q[ind] = d_q[ind];
            }
            __syncthreads();
            (void)d_workspace;
            // compute with NUM_TIMESTEPS as NUM_REPS for timing
            for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
                // anti-LICM: volatile reload of inputs each rep
                for(int _aopt_i = threadIdx.x + threadIdx.y*blockDim.x; _aopt_i < 7; _aopt_i += blockDim.x*blockDim.y){
                    reinterpret_cast<volatile T *>(s_q)[_aopt_i] = reinterpret_cast<const volatile T *>(d_q)[_aopt_i];
                }
                __syncthreads();
                // anti-LICM (1/2): stomp one input slot with `rep`
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
                    reinterpret_cast<volatile T *>(s_q)[rep % (7)] = static_cast<T>(rep);
                }
                // anti-LICM (2/2): feedback prev rep's d_Minv into s_q (true loop-carried dep)
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
                    T _aopt_fb1 = reinterpret_cast<const volatile T *>(d_Minv)[(rep + 0x3FF) & 0x3FF];
                    T _aopt_fb2 = reinterpret_cast<const volatile T *>(d_Minv)[(rep + 0x3FE) & 0x3FF];
                    T _aopt_fb3 = reinterpret_cast<const volatile T *>(d_Minv)[(rep + 0x3FD) & 0x3FF];
                    reinterpret_cast<volatile T *>(s_q)[(rep + 1) % (7)] += _aopt_fb1;
                    reinterpret_cast<volatile T *>(s_q)[(rep + 2) % (7)] += _aopt_fb2;
                    reinterpret_cast<volatile T *>(s_q)[(rep + 3) % (7)] += _aopt_fb3;
                }
                __syncthreads();
                load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
                minv_inner<T, true>(s_Minv, s_q, s_XImats, s_topology_helpers, s_temp, nullptr);
                __syncthreads();
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) { reinterpret_cast<volatile T *>(d_Minv)[rep & 1023] = reinterpret_cast<const volatile T *>(s_Minv)[rep & 7]; }
            }
            // save down to global
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 49; ind += blockDim.x*blockDim.y){
                d_Minv[ind] = s_Minv[ind];
            }
            __syncthreads();
        }
        else if constexpr (RESOURCE_TIER == TIER_MINIMAL) {
            // GRID shared arena layout
            //   T s_q[7]
            //   T s_Minv[49]
            //   T s_XImats[504]
            //   T s_temp[373]
            //   bytes s_linalg_smem[GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()]
            extern __shared__ __align__(16) unsigned char s_arena[];
            size_t s_arena_offset = 0;
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_q = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(7);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_Minv = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(49);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_XImats = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(504);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_temp = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(373);
            int *s_topology_helpers = nullptr;
            unsigned char *s_linalg_smem = nullptr;
            if (static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) > 0) {
                s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
                s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
                s_arena_offset += static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());
            }
            #ifdef GRID_CUDA_DEBUG_LAYOUT
            assert(s_arena_offset == grid_shared_arena_bytes<T>(933, 0, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()));
            #endif
            (void)s_arena_offset;
            // load to shared mem
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                s_q[ind] = d_q[ind];
            }
            __syncthreads();
            T *minv_d_workspace = reinterpret_cast<T *>(&d_workspace[GRID_MINV_F_WORKSPACE_OFFSET_BYTES<T>()]);
            // compute with NUM_TIMESTEPS as NUM_REPS for timing
            for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
                // anti-LICM: volatile reload of inputs each rep
                for(int _aopt_i = threadIdx.x + threadIdx.y*blockDim.x; _aopt_i < 7; _aopt_i += blockDim.x*blockDim.y){
                    reinterpret_cast<volatile T *>(s_q)[_aopt_i] = reinterpret_cast<const volatile T *>(d_q)[_aopt_i];
                }
                __syncthreads();
                // anti-LICM (1/2): stomp one input slot with `rep`
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
                    reinterpret_cast<volatile T *>(s_q)[rep % (7)] = static_cast<T>(rep);
                }
                // anti-LICM (2/2): feedback prev rep's d_Minv into s_q (true loop-carried dep)
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
                    T _aopt_fb1 = reinterpret_cast<const volatile T *>(d_Minv)[(rep + 0x3FF) & 0x3FF];
                    T _aopt_fb2 = reinterpret_cast<const volatile T *>(d_Minv)[(rep + 0x3FE) & 0x3FF];
                    T _aopt_fb3 = reinterpret_cast<const volatile T *>(d_Minv)[(rep + 0x3FD) & 0x3FF];
                    reinterpret_cast<volatile T *>(s_q)[(rep + 1) % (7)] += _aopt_fb1;
                    reinterpret_cast<volatile T *>(s_q)[(rep + 2) % (7)] += _aopt_fb2;
                    reinterpret_cast<volatile T *>(s_q)[(rep + 3) % (7)] += _aopt_fb3;
                }
                __syncthreads();
                load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
                minv_inner<T, false>(s_Minv, s_q, s_XImats, s_topology_helpers, s_temp, minv_d_workspace);
                __syncthreads();
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) { reinterpret_cast<volatile T *>(d_Minv)[rep & 1023] = reinterpret_cast<const volatile T *>(s_Minv)[rep & 7]; }
            }
            // save down to global
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 49; ind += blockDim.x*blockDim.y){
                d_Minv[ind] = s_Minv[ind];
            }
            __syncthreads();
        }
    }

    /**
     * Compute the inverse of the mass matrix
     *
     * Notes:
     *   Outputs a SYMMETRIC_UPPER triangular matrix for Minv
     *
     * @param d_Minv is a pointer to memory for the final result
     * @param d_workspace is the L2-pinned global spill buffer (used when Minv-F overflows smem)
     * @param d_q is the vector of joint positions
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER, bool MUJOCO_OUTPUT = false>
    __global__
    __launch_bounds__(tier_max_threads<RESOURCE_TIER>())
    void minv_kernel(T *d_Minv, unsigned char *d_workspace, const T *d_q, const int stride_q, const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS){
        if constexpr (RESOURCE_TIER == TIER_SHARED) {
            // GRID shared arena layout
            //   T s_q[7]
            //   T s_Minv[49]
            //   T s_XImats[504]
            //   T s_temp[667]
            //   bytes s_linalg_smem[GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()]
            extern __shared__ __align__(16) unsigned char s_arena[];
            size_t s_arena_offset = 0;
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_q = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(7);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_Minv = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(49);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_XImats = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(504);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_temp = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(667);
            int *s_topology_helpers = nullptr;
            unsigned char *s_linalg_smem = nullptr;
            if (static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) > 0) {
                s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
                s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
                s_arena_offset += static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());
            }
            #ifdef GRID_CUDA_DEBUG_LAYOUT
            assert(s_arena_offset == grid_shared_arena_bytes<T>(1227, 0, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()));
            #endif
            (void)s_arena_offset;
            for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
                // load to shared mem
                const T *d_q_k = &d_q[k*stride_q];
                for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                    s_q[ind] = d_q_k[ind];
                }
                __syncthreads();
                (void)d_workspace;
                // compute
                load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
                minv_inner<T, true>(s_Minv, s_q, s_XImats, s_topology_helpers, s_temp, nullptr);
                __syncthreads();
                // save down to global
                T *d_Minv_k = &d_Minv[k*49];
                for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 49; ind += blockDim.x*blockDim.y){
                    d_Minv_k[ind] = s_Minv[ind];
                }
                __syncthreads();
            }
        }
        else if constexpr (RESOURCE_TIER == TIER_LITE) {
            // GRID shared arena layout
            //   T s_q[7]
            //   T s_Minv[49]
            //   T s_XImats[504]
            //   T s_temp[667]
            //   bytes s_linalg_smem[GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()]
            extern __shared__ __align__(16) unsigned char s_arena[];
            size_t s_arena_offset = 0;
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_q = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(7);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_Minv = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(49);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_XImats = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(504);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_temp = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(667);
            int *s_topology_helpers = nullptr;
            unsigned char *s_linalg_smem = nullptr;
            if (static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) > 0) {
                s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
                s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
                s_arena_offset += static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());
            }
            #ifdef GRID_CUDA_DEBUG_LAYOUT
            assert(s_arena_offset == grid_shared_arena_bytes<T>(1227, 0, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()));
            #endif
            (void)s_arena_offset;
            for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
                // load to shared mem
                const T *d_q_k = &d_q[k*stride_q];
                for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                    s_q[ind] = d_q_k[ind];
                }
                __syncthreads();
                (void)d_workspace;
                // compute
                load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
                minv_inner<T, true>(s_Minv, s_q, s_XImats, s_topology_helpers, s_temp, nullptr);
                __syncthreads();
                // save down to global
                T *d_Minv_k = &d_Minv[k*49];
                for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 49; ind += blockDim.x*blockDim.y){
                    d_Minv_k[ind] = s_Minv[ind];
                }
                __syncthreads();
            }
        }
        else if constexpr (RESOURCE_TIER == TIER_MINIMAL) {
            // GRID shared arena layout
            //   T s_q[7]
            //   T s_Minv[49]
            //   T s_XImats[504]
            //   T s_temp[373]
            //   bytes s_linalg_smem[GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()]
            extern __shared__ __align__(16) unsigned char s_arena[];
            size_t s_arena_offset = 0;
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_q = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(7);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_Minv = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(49);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_XImats = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(504);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_temp = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(373);
            int *s_topology_helpers = nullptr;
            unsigned char *s_linalg_smem = nullptr;
            if (static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) > 0) {
                s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
                s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
                s_arena_offset += static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());
            }
            #ifdef GRID_CUDA_DEBUG_LAYOUT
            assert(s_arena_offset == grid_shared_arena_bytes<T>(933, 0, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()));
            #endif
            (void)s_arena_offset;
            for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
                // load to shared mem
                const T *d_q_k = &d_q[k*stride_q];
                for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                    s_q[ind] = d_q_k[ind];
                }
                __syncthreads();
                T *minv_d_workspace = reinterpret_cast<T *>(&d_workspace[grid_workspace_slot()*GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>() + GRID_MINV_F_WORKSPACE_OFFSET_BYTES<T>()]);
                // compute
                load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
                minv_inner<T, false>(s_Minv, s_q, s_XImats, s_topology_helpers, s_temp, minv_d_workspace);
                __syncthreads();
                // save down to global
                T *d_Minv_k = &d_Minv[k*49];
                for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 49; ind += blockDim.x*blockDim.y){
                    d_Minv_k[ind] = s_Minv[ind];
                }
                __syncthreads();
            }
        }
    }

    /**
     * Compute the inverse of the mass matrix
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_COMPRESSED_MEM = false, gridDataKind KIND = GRID_DATA_ALL, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>
    __host__
    void minv(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps,
                     const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_DYNAMICS, "minv requires all-data or dynamics gridData");
        // start code with memory transfer
        int stride_q;
        if (USE_COMPRESSED_MEM) {stride_q = NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q,hd_data->h_q,stride_q*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        else {stride_q = 3*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        gpuErrchkKernel();
        // then call the kernel
        gpuErrchk(grid_check_dynamic_shared_memory_bytes("minv", MINV_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()));
        const int _grid_ws_n = (hd_data->workspace_timestep_slots > 0 && hd_data->workspace_timestep_slots < num_timesteps) ? hd_data->workspace_timestep_slots : num_timesteps;
        dim3 _ws_grid = block_dimms;
        if ((int)(_ws_grid.x*_ws_grid.y*_ws_grid.z) > _grid_ws_n) { _ws_grid = dim3(_grid_ws_n,1,1); }
        dim3 _grid_thr_clamped_13 = grid_host_clamp_threads((const void*)&minv_kernel<T, RESOURCE_TIER>, thread_dimms);
        if (USE_COMPRESSED_MEM) {minv_kernel<T, RESOURCE_TIER><<<_ws_grid,_grid_thr_clamped_13,MINV_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()>>>(hd_data->d_Minv,hd_data->d_workspace,hd_data->d_q,stride_q,d_robotModel,num_timesteps);}
        else                    {minv_kernel<T, RESOURCE_TIER><<<_ws_grid,_grid_thr_clamped_13,MINV_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()>>>(hd_data->d_Minv,hd_data->d_workspace,hd_data->d_q_qd_u,stride_q,d_robotModel,num_timesteps);}
        gpuErrchkKernel();
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_Minv,hd_data->d_Minv,NUM_VEL*NUM_VEL*num_timesteps*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchkKernel();
    }

    /**
     * Compute the inverse of the mass matrix
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_COMPRESSED_MEM = false, gridDataKind KIND = GRID_DATA_ALL, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>
    __host__
    void minv_single_timing(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps,
                                   const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_DYNAMICS, "minv requires all-data or dynamics gridData");
        // start code with memory transfer
        int stride_q;
        if (USE_COMPRESSED_MEM) {stride_q = NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q,hd_data->h_q,stride_q*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        else {stride_q = 3*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        gpuErrchkKernel();
        // then call the kernel
        gpuErrchk(grid_check_dynamic_shared_memory_bytes("minv", MINV_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()));
        struct timespec start, end; clock_gettime(CLOCK_MONOTONIC,&start);
        dim3 _grid_thr_clamped_14 = grid_host_clamp_threads((const void*)&minv_kernel_single_timing<T, RESOURCE_TIER>, thread_dimms);
        if (USE_COMPRESSED_MEM) {minv_kernel_single_timing<T, RESOURCE_TIER><<<block_dimms,_grid_thr_clamped_14,MINV_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()>>>(hd_data->d_Minv,hd_data->d_workspace,hd_data->d_q,stride_q,d_robotModel,num_timesteps);}
        else                    {minv_kernel_single_timing<T, RESOURCE_TIER><<<block_dimms,_grid_thr_clamped_14,MINV_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()>>>(hd_data->d_Minv,hd_data->d_workspace,hd_data->d_q_qd_u,stride_q,d_robotModel,num_timesteps);}
        gpuErrchkKernel();
        clock_gettime(CLOCK_MONOTONIC,&end);
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_Minv,hd_data->d_Minv,NUM_VEL*NUM_VEL*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchkKernel();
        printf("Single Call MINV %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(num_timesteps));
    }

    /**
     * Compute the inverse of the mass matrix
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_COMPRESSED_MEM = false, gridDataKind KIND = GRID_DATA_ALL, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>
    __host__
    void minv_compute_only(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps,
                                  const dim3 block_dimms, const dim3 thread_dimms) {
        static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_DYNAMICS, "minv requires all-data or dynamics gridData");
        int stride_q = USE_COMPRESSED_MEM ? NUM_JOINTS: 3*NUM_JOINTS;
        // then call the kernel
        gpuErrchk(grid_check_dynamic_shared_memory_bytes("minv", MINV_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()));
        const int _grid_ws_n = (hd_data->workspace_timestep_slots > 0 && hd_data->workspace_timestep_slots < num_timesteps) ? hd_data->workspace_timestep_slots : num_timesteps;
        dim3 _ws_grid = block_dimms;
        if ((int)(_ws_grid.x*_ws_grid.y*_ws_grid.z) > _grid_ws_n) { _ws_grid = dim3(_grid_ws_n,1,1); }
        dim3 _grid_thr_clamped_15 = grid_host_clamp_threads((const void*)&minv_kernel<T, RESOURCE_TIER>, thread_dimms);
        if (USE_COMPRESSED_MEM) {minv_kernel<T, RESOURCE_TIER><<<_ws_grid,_grid_thr_clamped_15,MINV_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()>>>(hd_data->d_Minv,hd_data->d_workspace,hd_data->d_q,stride_q,d_robotModel,num_timesteps);}
        else                    {minv_kernel<T, RESOURCE_TIER><<<_ws_grid,_grid_thr_clamped_15,MINV_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()>>>(hd_data->d_Minv,hd_data->d_workspace,hd_data->d_q_qd_u,stride_q,d_robotModel,num_timesteps);}
        gpuErrchkKernel();
    }

    /**
     * Finish the forward dynamics computation with qdd = Minv*(u-c)
     *
     * Notes:
     *   Assumes s_Minv and s_c are already computed
     *   Does not internally sync the thread group, so it should be called after all threads have finished computing their values
     *   CALLER CONTRACT (post): also does not sync AFTER its s_qdd writes -- a hand-composed caller MUST __syncthreads() before any thread READS s_qdd or reuses the s_c/s_Minv storage (GRiD's own generated compositions do; racecheck flags the missing sync as a fd_finish-write vs downstream-read hazard, e.g. vs inverse_dynamics_inner_vaf)
     *
     * @param s_qdd is a pointer to memory for the final result
     * @param s_u is the vector of joint input torques
     * @param s_c is the bias vector
     * @param s_Minv is the inverse mass matrix
     */
    template <typename T>
    __device__
    void forward_dynamics_finish(T *s_qdd, const T *s_u, const T *s_c, const T *s_Minv) {
        for(int row = threadIdx.x + threadIdx.y*blockDim.x; row < 7; row += blockDim.x*blockDim.y){
            T val = static_cast<T>(0);
            for(int col = 0; col < 7; col++) {
                // account for the fact that Minv is an SYMMETRIC_UPPER triangular matrix
                int index = (row <= col) * (col * 7 + row) + (row > col) * (row * 7 + col);
                val += s_Minv[index] * (s_u[col] - s_c[col]);
            }
            s_qdd[row] = val;
        }
    }

    /**
     * Computes forward dynamics
     *
     * Notes:
     *   CALLER CONTRACT (direct *_inner callers): s_XImats must ALREADY be populated for the current s_q -- the inner READS but never writes it. Call load_update_XImats_helpers(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp) then __syncthreads() first, or just call forward_dynamics_device (which does this for you). Skipping it reads uninitialized shared -> NaN (race-clean, initcheck-fixable).
     *   CALLER CONTRACT (sizing): s_temp MUST be FD_INNER_SMEM_BYTES<T, MINV_F_IN_SMEM>() bytes. At MINV_F_IN_SMEM=true the 6*NV*NV Minv-F band lives in the TAIL of s_temp (the macro includes it); d_workspace is 0/nullptr but that does NOT mean the band is free -- it just moved into s_temp. Under-sizing s_temp (e.g. reusing a fewer-DoF constant) makes the inner read its own never-written band -> NaN, and the failure is DoF-specific because the band scales as 6*NV*NV.
     *   Does not internally sync the thread group, so it should be called after all threads have finished computing their values
     *   Inner-controlled placement: MINV_F_IN_SMEM selects where the internal Minv 6*NV*NV F-region lives (s_temp tail vs d_workspace). Decided here; caller sizes both arenas from FD_INNER_*_BYTES and hands both pointers in.
     *
     * @param s_qdd is a pointer to memory for the final result
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param s_u is the vector of joint input torques
     * @param s_temp is the (shared) scratch; size FD_INNER_SMEM_BYTES<T, MINV_F_IN_SMEM>()
     * @param d_workspace is the global scratch; size FD_INNER_WORKSPACE_BYTES<T, MINV_F_IN_SMEM>() (= 6*NV*NV when !MINV_F_IN_SMEM, else 0). Pass nullptr when MINV_F_IN_SMEM
     * @param s_XImats is the (shared) memory holding the updated XI matricies for the given s_q
     * @param s_topology_helpers is the (shared) memory location for the topology_helpers (nullptr/unused for serial chains with identical Ss)
     * @param d_f_ext is the (optional) GLOBAL external forces, body-major 6*NUM_BODIES local-frame, or nullptr
     * @param gravity is the gravity constant
     */
    template <typename T, bool MINV_F_IN_SMEM = true>
    __device__
    void forward_dynamics_inner(T *s_qdd, const T *s_q, const T *s_qd, const T *s_u, T *s_XImats, int *s_topology_helpers, T *s_temp, T *d_workspace, T *d_f_ext, const T gravity) {
        minv_inner<T, MINV_F_IN_SMEM>(s_temp, s_q, s_XImats, s_topology_helpers, &s_temp[49], d_workspace);
        inverse_dynamics_inner<T>(&s_temp[49], &s_temp[56], s_q, s_qd, s_XImats, s_topology_helpers, &s_temp[182], d_f_ext, gravity);
        forward_dynamics_finish<T>(s_qdd, s_u, &s_temp[49], s_temp);
    }

    /**
     * Computes forward dynamics
     *
     * Notes:
     *   Inline-CUDA users: at TIER_LITE/TIER_MINIMAL the whole FD inner scratch (~716*sizeof(T) bytes) moves from shared memory to d_workspace, freeing smem for the caller's outer kernel.
     *   The inner-temp arena holds the (6*NUM_VEL*NUM_VEL) Minv-F band at its tail, so routing the whole arena to d_workspace also spills the F-band; this is the device-path analog of the kernel's MINV_F_IN_SMEM lever (which surgically spills only F).
     *
     * @param s_qdd is a pointer to memory for the final result
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param s_u is the vector of joint input torques
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param d_f_ext is the (optional) GLOBAL external forces, body-major 6*NUM_BODIES local-frame, or nullptr
     * @param gravity is the gravity constant
     * @param d_workspace is the global scratch buffer; size FORWARD_DYNAMICS_DEVICE_INLINE_WORKSPACE_BYTES<T, RESOURCE_TIER>() bytes (= 0 at TIER_SHARED, 716*sizeof(T) at TIER_LITE+). Pass nullptr at TIER_SHARED
     */
    template <typename T, int RESOURCE_TIER = TIER_SHARED>
    __device__
    void forward_dynamics_device(T *s_qdd, const T *s_q, const T *s_qd, const T *s_u, const robotModel<T> *d_robotModel, T *d_f_ext, const T gravity, T *d_workspace = nullptr) {
        // GRID shared arena layout
        //   T s_XImats[504]
        //   T s_temp[716] (TIER_SHARED only; LITE/MINIMAL route to d_workspace)
        //   bytes s_linalg_smem[GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()]
        extern __shared__ __align__(16) unsigned char s_arena[];
        size_t s_arena_offset = 0;
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_XImats = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(504);
        T *s_temp;
        if constexpr (RESOURCE_TIER == TIER_SHARED) {
            (void)d_workspace;
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            s_temp = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(716);
        }
        else {
            s_temp = d_workspace;
        }
        int *s_topology_helpers = nullptr;
        unsigned char *s_linalg_smem = nullptr;
        if (static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) > 0) {
            s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
            s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
            s_arena_offset += static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());
        }
        #ifdef GRID_CUDA_DEBUG_LAYOUT
        if constexpr (RESOURCE_TIER == TIER_SHARED) {
            assert(s_arena_offset == grid_shared_arena_bytes<T>(1220, 0, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()));
        }
        else {
            assert(s_arena_offset == grid_shared_arena_bytes<T>(504, 0, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()));
        }
        #endif
        (void)s_arena_offset;
        load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
        forward_dynamics_inner<T, true>(s_qdd, s_q, s_qd, s_u, s_XImats, s_topology_helpers, s_temp, nullptr, d_f_ext, gravity);
    }

    /**
     * Computes forward dynamics
     *
     * @param d_qdd is a pointer to memory for the final result
     * @param d_workspace is the L2-pinned global spill buffer (used when Minv-F overflows smem)
     * @param d_q_qd_u is the vector of joint positions, velocities, and input torques
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param d_f_ext is the (optional) GLOBAL external forces, body-major 6*NUM_BODIES local-frame, or nullptr
     * @param gravity is the gravity constant
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER, bool MUJOCO_OUTPUT = false>
    __global__
    __launch_bounds__(tier_max_threads<RESOURCE_TIER>())
    void forward_dynamics_kernel_single_timing(T *d_qdd, unsigned char *d_workspace, const T *d_q_qd_u, const int stride_q_qd_u, T *d_f_ext, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        if constexpr (RESOURCE_TIER == TIER_SHARED) {
            // GRID shared arena layout
            //   T s_q_qd_u[21]
            //   T s_qdd[7]
            //   T s_XImats[504]
            //   T s_temp[716]
            //   bytes s_linalg_smem[GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()]
            extern __shared__ __align__(16) unsigned char s_arena[];
            size_t s_arena_offset = 0;
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_q_qd_u = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(21);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_qdd = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(7);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_XImats = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(504);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_temp = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(716);
            int *s_topology_helpers = nullptr;
            unsigned char *s_linalg_smem = nullptr;
            if (static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) > 0) {
                s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
                s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
                s_arena_offset += static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());
            }
            #ifdef GRID_CUDA_DEBUG_LAYOUT
            assert(s_arena_offset == grid_shared_arena_bytes<T>(1248, 0, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()));
            #endif
            (void)s_arena_offset;
            T *s_q = s_q_qd_u; T *s_qd = &s_q_qd_u[7]; T *s_u = &s_q_qd_u[14];
            // load to shared mem
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 21; ind += blockDim.x*blockDim.y){
                s_q_qd_u[ind] = d_q_qd_u[ind];
            }
            __syncthreads();
            (void)d_workspace;
            // compute with NUM_TIMESTEPS as NUM_REPS for timing
            for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
                // anti-LICM: volatile reload of inputs each rep
                for(int _aopt_i = threadIdx.x + threadIdx.y*blockDim.x; _aopt_i < 21; _aopt_i += blockDim.x*blockDim.y){
                    reinterpret_cast<volatile T *>(s_q_qd_u)[_aopt_i] = reinterpret_cast<const volatile T *>(d_q_qd_u)[_aopt_i];
                }
                __syncthreads();
                // anti-LICM (1/2): stomp one input slot with `rep`
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
                    reinterpret_cast<volatile T *>(s_q_qd_u)[rep % (21)] = static_cast<T>(rep);
                }
                // anti-LICM (2/2): feedback prev rep's d_qdd into s_q_qd_u (true loop-carried dep)
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
                    T _aopt_fb1 = reinterpret_cast<const volatile T *>(d_qdd)[(rep + 0x3FF) & 0x3FF];
                    T _aopt_fb2 = reinterpret_cast<const volatile T *>(d_qdd)[(rep + 0x3FE) & 0x3FF];
                    T _aopt_fb3 = reinterpret_cast<const volatile T *>(d_qdd)[(rep + 0x3FD) & 0x3FF];
                    reinterpret_cast<volatile T *>(s_q_qd_u)[(rep + 1) % (21)] += _aopt_fb1;
                    reinterpret_cast<volatile T *>(s_q_qd_u)[(rep + 2) % (21)] += _aopt_fb2;
                    reinterpret_cast<volatile T *>(s_q_qd_u)[(rep + 3) % (21)] += _aopt_fb3;
                }
                __syncthreads();
                load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
                forward_dynamics_inner<T, true>(s_qdd, s_q, s_qd, s_u, s_XImats, s_topology_helpers, s_temp, nullptr, d_f_ext, gravity);
                __syncthreads();
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) { reinterpret_cast<volatile T *>(d_qdd)[rep & 1023] = reinterpret_cast<const volatile T *>(s_qdd)[rep & 7]; }
            }
            // save down to global
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                d_qdd[ind] = s_qdd[ind];
            }
            __syncthreads();
        }
        else if constexpr (RESOURCE_TIER == TIER_LITE) {
            // GRID shared arena layout
            //   T s_q_qd_u[21]
            //   T s_qdd[7]
            //   T s_XImats[504]
            //   T s_temp[716]
            //   bytes s_linalg_smem[GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()]
            extern __shared__ __align__(16) unsigned char s_arena[];
            size_t s_arena_offset = 0;
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_q_qd_u = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(21);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_qdd = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(7);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_XImats = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(504);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_temp = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(716);
            int *s_topology_helpers = nullptr;
            unsigned char *s_linalg_smem = nullptr;
            if (static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) > 0) {
                s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
                s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
                s_arena_offset += static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());
            }
            #ifdef GRID_CUDA_DEBUG_LAYOUT
            assert(s_arena_offset == grid_shared_arena_bytes<T>(1248, 0, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()));
            #endif
            (void)s_arena_offset;
            T *s_q = s_q_qd_u; T *s_qd = &s_q_qd_u[7]; T *s_u = &s_q_qd_u[14];
            // load to shared mem
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 21; ind += blockDim.x*blockDim.y){
                s_q_qd_u[ind] = d_q_qd_u[ind];
            }
            __syncthreads();
            (void)d_workspace;
            // compute with NUM_TIMESTEPS as NUM_REPS for timing
            for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
                // anti-LICM: volatile reload of inputs each rep
                for(int _aopt_i = threadIdx.x + threadIdx.y*blockDim.x; _aopt_i < 21; _aopt_i += blockDim.x*blockDim.y){
                    reinterpret_cast<volatile T *>(s_q_qd_u)[_aopt_i] = reinterpret_cast<const volatile T *>(d_q_qd_u)[_aopt_i];
                }
                __syncthreads();
                // anti-LICM (1/2): stomp one input slot with `rep`
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
                    reinterpret_cast<volatile T *>(s_q_qd_u)[rep % (21)] = static_cast<T>(rep);
                }
                // anti-LICM (2/2): feedback prev rep's d_qdd into s_q_qd_u (true loop-carried dep)
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
                    T _aopt_fb1 = reinterpret_cast<const volatile T *>(d_qdd)[(rep + 0x3FF) & 0x3FF];
                    T _aopt_fb2 = reinterpret_cast<const volatile T *>(d_qdd)[(rep + 0x3FE) & 0x3FF];
                    T _aopt_fb3 = reinterpret_cast<const volatile T *>(d_qdd)[(rep + 0x3FD) & 0x3FF];
                    reinterpret_cast<volatile T *>(s_q_qd_u)[(rep + 1) % (21)] += _aopt_fb1;
                    reinterpret_cast<volatile T *>(s_q_qd_u)[(rep + 2) % (21)] += _aopt_fb2;
                    reinterpret_cast<volatile T *>(s_q_qd_u)[(rep + 3) % (21)] += _aopt_fb3;
                }
                __syncthreads();
                load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
                forward_dynamics_inner<T, true>(s_qdd, s_q, s_qd, s_u, s_XImats, s_topology_helpers, s_temp, nullptr, d_f_ext, gravity);
                __syncthreads();
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) { reinterpret_cast<volatile T *>(d_qdd)[rep & 1023] = reinterpret_cast<const volatile T *>(s_qdd)[rep & 7]; }
            }
            // save down to global
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                d_qdd[ind] = s_qdd[ind];
            }
            __syncthreads();
        }
        else if constexpr (RESOURCE_TIER == TIER_MINIMAL) {
            // GRID shared arena layout
            //   T s_q_qd_u[21]
            //   T s_qdd[7]
            //   T s_XImats[504]
            //   T s_temp[422]
            //   bytes s_linalg_smem[GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()]
            extern __shared__ __align__(16) unsigned char s_arena[];
            size_t s_arena_offset = 0;
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_q_qd_u = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(21);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_qdd = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(7);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_XImats = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(504);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_temp = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(422);
            int *s_topology_helpers = nullptr;
            unsigned char *s_linalg_smem = nullptr;
            if (static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) > 0) {
                s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
                s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
                s_arena_offset += static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());
            }
            #ifdef GRID_CUDA_DEBUG_LAYOUT
            assert(s_arena_offset == grid_shared_arena_bytes<T>(954, 0, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()));
            #endif
            (void)s_arena_offset;
            T *s_q = s_q_qd_u; T *s_qd = &s_q_qd_u[7]; T *s_u = &s_q_qd_u[14];
            // load to shared mem
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 21; ind += blockDim.x*blockDim.y){
                s_q_qd_u[ind] = d_q_qd_u[ind];
            }
            __syncthreads();
            T *fd_d_workspace = reinterpret_cast<T *>(&d_workspace[GRID_MINV_F_WORKSPACE_OFFSET_BYTES<T>()]);
            // compute with NUM_TIMESTEPS as NUM_REPS for timing
            for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
                // anti-LICM: volatile reload of inputs each rep
                for(int _aopt_i = threadIdx.x + threadIdx.y*blockDim.x; _aopt_i < 21; _aopt_i += blockDim.x*blockDim.y){
                    reinterpret_cast<volatile T *>(s_q_qd_u)[_aopt_i] = reinterpret_cast<const volatile T *>(d_q_qd_u)[_aopt_i];
                }
                __syncthreads();
                // anti-LICM (1/2): stomp one input slot with `rep`
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
                    reinterpret_cast<volatile T *>(s_q_qd_u)[rep % (21)] = static_cast<T>(rep);
                }
                // anti-LICM (2/2): feedback prev rep's d_qdd into s_q_qd_u (true loop-carried dep)
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
                    T _aopt_fb1 = reinterpret_cast<const volatile T *>(d_qdd)[(rep + 0x3FF) & 0x3FF];
                    T _aopt_fb2 = reinterpret_cast<const volatile T *>(d_qdd)[(rep + 0x3FE) & 0x3FF];
                    T _aopt_fb3 = reinterpret_cast<const volatile T *>(d_qdd)[(rep + 0x3FD) & 0x3FF];
                    reinterpret_cast<volatile T *>(s_q_qd_u)[(rep + 1) % (21)] += _aopt_fb1;
                    reinterpret_cast<volatile T *>(s_q_qd_u)[(rep + 2) % (21)] += _aopt_fb2;
                    reinterpret_cast<volatile T *>(s_q_qd_u)[(rep + 3) % (21)] += _aopt_fb3;
                }
                __syncthreads();
                load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
                forward_dynamics_inner<T, false>(s_qdd, s_q, s_qd, s_u, s_XImats, s_topology_helpers, s_temp, fd_d_workspace, d_f_ext, gravity);
                __syncthreads();
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) { reinterpret_cast<volatile T *>(d_qdd)[rep & 1023] = reinterpret_cast<const volatile T *>(s_qdd)[rep & 7]; }
            }
            // save down to global
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                d_qdd[ind] = s_qdd[ind];
            }
            __syncthreads();
        }
    }

    /**
     * Computes forward dynamics
     *
     * @param d_qdd is a pointer to memory for the final result
     * @param d_workspace is the L2-pinned global spill buffer (used when Minv-F overflows smem)
     * @param d_q_qd_u is the vector of joint positions, velocities, and input torques
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param d_f_ext is the (optional) GLOBAL external forces, body-major 6*NUM_BODIES local-frame, or nullptr
     * @param gravity is the gravity constant
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER, bool MUJOCO_OUTPUT = false>
    __global__
    __launch_bounds__(tier_max_threads<RESOURCE_TIER>())
    void forward_dynamics_kernel(T *d_qdd, unsigned char *d_workspace, const T *d_q_qd_u, const int stride_q_qd_u, T *d_f_ext, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        if constexpr (RESOURCE_TIER == TIER_SHARED) {
            // GRID shared arena layout
            //   T s_q_qd_u[21]
            //   T s_qdd[7]
            //   T s_XImats[504]
            //   T s_temp[716]
            //   bytes s_linalg_smem[GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()]
            extern __shared__ __align__(16) unsigned char s_arena[];
            size_t s_arena_offset = 0;
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_q_qd_u = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(21);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_qdd = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(7);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_XImats = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(504);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_temp = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(716);
            int *s_topology_helpers = nullptr;
            unsigned char *s_linalg_smem = nullptr;
            if (static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) > 0) {
                s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
                s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
                s_arena_offset += static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());
            }
            #ifdef GRID_CUDA_DEBUG_LAYOUT
            assert(s_arena_offset == grid_shared_arena_bytes<T>(1248, 0, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()));
            #endif
            (void)s_arena_offset;
            T *s_q = s_q_qd_u; T *s_qd = &s_q_qd_u[7]; T *s_u = &s_q_qd_u[14];
            for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
                // load to shared mem
                const T *d_q_qd_u_k = &d_q_qd_u[k*stride_q_qd_u];
                for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 21; ind += blockDim.x*blockDim.y){
                    s_q_qd_u[ind] = d_q_qd_u_k[ind];
                }
                __syncthreads();
                (void)d_workspace;
                // compute
                load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
                forward_dynamics_inner<T, true>(s_qdd, s_q, s_qd, s_u, s_XImats, s_topology_helpers, s_temp, nullptr, d_f_ext, gravity);
                __syncthreads();
                // save down to global
                T *d_qdd_k = &d_qdd[k*7];
                for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                    d_qdd_k[ind] = s_qdd[ind];
                }
                __syncthreads();
            }
        }
        else if constexpr (RESOURCE_TIER == TIER_LITE) {
            // GRID shared arena layout
            //   T s_q_qd_u[21]
            //   T s_qdd[7]
            //   T s_XImats[504]
            //   T s_temp[716]
            //   bytes s_linalg_smem[GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()]
            extern __shared__ __align__(16) unsigned char s_arena[];
            size_t s_arena_offset = 0;
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_q_qd_u = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(21);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_qdd = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(7);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_XImats = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(504);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_temp = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(716);
            int *s_topology_helpers = nullptr;
            unsigned char *s_linalg_smem = nullptr;
            if (static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) > 0) {
                s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
                s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
                s_arena_offset += static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());
            }
            #ifdef GRID_CUDA_DEBUG_LAYOUT
            assert(s_arena_offset == grid_shared_arena_bytes<T>(1248, 0, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()));
            #endif
            (void)s_arena_offset;
            T *s_q = s_q_qd_u; T *s_qd = &s_q_qd_u[7]; T *s_u = &s_q_qd_u[14];
            for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
                // load to shared mem
                const T *d_q_qd_u_k = &d_q_qd_u[k*stride_q_qd_u];
                for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 21; ind += blockDim.x*blockDim.y){
                    s_q_qd_u[ind] = d_q_qd_u_k[ind];
                }
                __syncthreads();
                (void)d_workspace;
                // compute
                load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
                forward_dynamics_inner<T, true>(s_qdd, s_q, s_qd, s_u, s_XImats, s_topology_helpers, s_temp, nullptr, d_f_ext, gravity);
                __syncthreads();
                // save down to global
                T *d_qdd_k = &d_qdd[k*7];
                for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                    d_qdd_k[ind] = s_qdd[ind];
                }
                __syncthreads();
            }
        }
        else if constexpr (RESOURCE_TIER == TIER_MINIMAL) {
            // GRID shared arena layout
            //   T s_q_qd_u[21]
            //   T s_qdd[7]
            //   T s_XImats[504]
            //   T s_temp[422]
            //   bytes s_linalg_smem[GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()]
            extern __shared__ __align__(16) unsigned char s_arena[];
            size_t s_arena_offset = 0;
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_q_qd_u = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(21);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_qdd = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(7);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_XImats = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(504);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_temp = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(422);
            int *s_topology_helpers = nullptr;
            unsigned char *s_linalg_smem = nullptr;
            if (static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) > 0) {
                s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
                s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
                s_arena_offset += static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());
            }
            #ifdef GRID_CUDA_DEBUG_LAYOUT
            assert(s_arena_offset == grid_shared_arena_bytes<T>(954, 0, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()));
            #endif
            (void)s_arena_offset;
            T *s_q = s_q_qd_u; T *s_qd = &s_q_qd_u[7]; T *s_u = &s_q_qd_u[14];
            for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
                // load to shared mem
                const T *d_q_qd_u_k = &d_q_qd_u[k*stride_q_qd_u];
                for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 21; ind += blockDim.x*blockDim.y){
                    s_q_qd_u[ind] = d_q_qd_u_k[ind];
                }
                __syncthreads();
                T *fd_d_workspace = reinterpret_cast<T *>(&d_workspace[grid_workspace_slot()*GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>() + GRID_MINV_F_WORKSPACE_OFFSET_BYTES<T>()]);
                // compute
                load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
                forward_dynamics_inner<T, false>(s_qdd, s_q, s_qd, s_u, s_XImats, s_topology_helpers, s_temp, fd_d_workspace, d_f_ext, gravity);
                __syncthreads();
                // save down to global
                T *d_qdd_k = &d_qdd[k*7];
                for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                    d_qdd_k[ind] = s_qdd[ind];
                }
                __syncthreads();
            }
        }
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, gridDataKind KIND = GRID_DATA_ALL, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>
    __host__
    void forward_dynamics(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                          const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_DYNAMICS, "forward_dynamics requires all-data or dynamics gridData");
        int stride_q_qd_u = 3*NUM_JOINTS;
        // start code with memory transfer
        gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd_u*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[0]));
        gpuErrchkKernel();
        // then call the kernel
        gpuErrchk(grid_check_dynamic_shared_memory_bytes("forward_dynamics", FORWARD_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()));
        const int _grid_ws_n = (hd_data->workspace_timestep_slots > 0 && hd_data->workspace_timestep_slots < num_timesteps) ? hd_data->workspace_timestep_slots : num_timesteps;
        dim3 _ws_grid = block_dimms;
        if ((int)(_ws_grid.x*_ws_grid.y*_ws_grid.z) > _grid_ws_n) { _ws_grid = dim3(_grid_ws_n,1,1); }
        dim3 _grid_thr_clamped_16 = grid_host_clamp_threads((const void*)&forward_dynamics_kernel<T, RESOURCE_TIER>, thread_dimms);
        forward_dynamics_kernel<T, RESOURCE_TIER><<<_ws_grid,_grid_thr_clamped_16,FORWARD_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()>>>(hd_data->d_qdd,hd_data->d_workspace,hd_data->d_q_qd_u,stride_q_qd_u,hd_data->d_f_ext,d_robotModel,gravity,num_timesteps);
        gpuErrchkKernel();
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_qdd,hd_data->d_qdd,NUM_JOINTS*num_timesteps*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchkKernel();
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, gridDataKind KIND = GRID_DATA_ALL, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>
    __host__
    void forward_dynamics_single_timing(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                                        const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_DYNAMICS, "forward_dynamics requires all-data or dynamics gridData");
        int stride_q_qd_u = 3*NUM_JOINTS;
        // start code with memory transfer
        gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd_u*sizeof(T),cudaMemcpyHostToDevice,streams[0]));
        gpuErrchkKernel();
        // then call the kernel
        gpuErrchk(grid_check_dynamic_shared_memory_bytes("forward_dynamics", FORWARD_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()));
        struct timespec start, end; clock_gettime(CLOCK_MONOTONIC,&start);
        dim3 _grid_thr_clamped_17 = grid_host_clamp_threads((const void*)&forward_dynamics_kernel_single_timing<T, RESOURCE_TIER>, thread_dimms);
        forward_dynamics_kernel_single_timing<T, RESOURCE_TIER><<<block_dimms,_grid_thr_clamped_17,FORWARD_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()>>>(hd_data->d_qdd,hd_data->d_workspace,hd_data->d_q_qd_u,stride_q_qd_u,hd_data->d_f_ext,d_robotModel,gravity,num_timesteps);
        gpuErrchkKernel();
        clock_gettime(CLOCK_MONOTONIC,&end);
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_qdd,hd_data->d_qdd,NUM_JOINTS*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchkKernel();
        printf("Single Call FORWARD_DYNAMICS %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(num_timesteps));
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, gridDataKind KIND = GRID_DATA_ALL, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>
    __host__
    void forward_dynamics_compute_only(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                                       const dim3 block_dimms, const dim3 thread_dimms) {
        static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_DYNAMICS, "forward_dynamics requires all-data or dynamics gridData");
        int stride_q_qd_u = 3*NUM_JOINTS;
        // then call the kernel
        gpuErrchk(grid_check_dynamic_shared_memory_bytes("forward_dynamics", FORWARD_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()));
        const int _grid_ws_n = (hd_data->workspace_timestep_slots > 0 && hd_data->workspace_timestep_slots < num_timesteps) ? hd_data->workspace_timestep_slots : num_timesteps;
        dim3 _ws_grid = block_dimms;
        if ((int)(_ws_grid.x*_ws_grid.y*_ws_grid.z) > _grid_ws_n) { _ws_grid = dim3(_grid_ws_n,1,1); }
        dim3 _grid_thr_clamped_18 = grid_host_clamp_threads((const void*)&forward_dynamics_kernel<T, RESOURCE_TIER>, thread_dimms);
        forward_dynamics_kernel<T, RESOURCE_TIER><<<_ws_grid,_grid_thr_clamped_18,FORWARD_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()>>>(hd_data->d_qdd,hd_data->d_workspace,hd_data->d_q_qd_u,stride_q_qd_u,hd_data->d_f_ext,d_robotModel,gravity,num_timesteps);
        gpuErrchkKernel();
    }

    /**
     * Computes the gradient of inverse dynamics
     *
     * Notes:
     *   Assumes s_XImats is updated already for the current s_q
     *   This is the inverse_dynamics_gradient band sub-inner (the stable surface composed by forward_dynamics_gradient / integrator_gradient). It does NOT own s_temp placement; the USE_DA_DF_SPILL band selectively spills its da_dq..fxvi band to d_temp_spill via grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>. The whole-pool placement is owned by the wrapping inverse_dynamics_gradient_device.
     *
     * @param s_dc_du is a pointer to memory for the final result of size 2*NUM_VEL*NUM_VEL = 98
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param s_vaf are the helper intermediate variables computed by inverse_dynamics
     * @param s_XImats is the (shared) memory holding the updated XI matricies for the given s_q
     * @param s_topology_helpers is the (shared) memory location for the topology_helpers (nullptr/unused for serial chains with identical Ss)
     * @param s_temp is a pointer to helper shared memory of size 66*NUM_JOINTS + 6*sparse_dv,da,df_col_needs = 1722
     * @param gravity is the gravity constant
     */
    template <typename T, bool USE_DA_DF_SPILL = false>
    __device__
    void inverse_dynamics_gradient_inner(T *s_dc_du, const T *s_q, const T *s_qd, const T *s_vaf, T *s_XImats, int *s_topology_helpers, T *s_temp, T *d_temp_spill, const T gravity) {
        //
        // dv and da need 28 cols per dq,dqd
        // df needs 49 cols per dq,dqd
        //    out of a possible 49 cols per dq,dqd
        // Gradients are stored compactly as dv_i/dq_[0...a], dv_i+1/dq_[0...b], etc
        //    where a and b are the needed number of columns
        //
        // Temp memory offsets are as follows:
        // T *s_dv_dq = grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, 0); T *s_dv_dqd = grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, 168); T *s_da_dq = grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, 336);
        // T *s_da_dqd = grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, 504); T *s_df_dq = grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, 672); T *s_df_dqd = grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, 966);
        // T *s_FxvI = grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, 1260); T *s_MxXv = grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, 1512); T *s_MxXa = grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, 1554);
        // T *s_Mxv = grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, 1596); T *s_Mxf = grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, 1638); T *s_Iv = grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, 1680);
        //
        // Initial Temp Comps
        //
        // First compute Imat*v and Xmat*v_parent, Xmat*a_parent (store in FxvI for now)
        // Note that if jid_parent == -1 then v_parent = 0 and a_parent = gravity
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 126; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int jid = col % 7; int jid6 = 6*jid;
            bool parentIsBase = (jid-1) == -1;
            bool comp1 = col < 7; bool comp3 = col >= 14;
            int XIOffset  =  comp1 * 252 + 6*jid6 + row; // rowCol of I (comp1) or X (comp 2 and 3)
            int vaOffset  = comp1 * jid6 + !comp1 * 6*(jid-1) + comp3 * 42; // v_i (comp1) or va_parent (comp 2 and 3)
            int dstOffset = comp1 * 1680 + !comp1 * 1260 + comp3 * 42 + jid6 + row; // rowCol of dst
            (*grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, dstOffset)) = (parentIsBase && !comp1) ? comp3 * -s_XImats[XIOffset + 30] * gravity :
                                                           dot_prod<T,6,6,1>(&s_XImats[XIOffset],&s_vaf[vaOffset]);
        }
        __syncthreads();
        // Then compute Mx(Xv), Mx(Xa), Mx(v), Mx(f)
        for(int col = threadIdx.x + threadIdx.y*blockDim.x; col < 28; col += blockDim.x*blockDim.y){
            int dof_id = col / 4; int selector = col % 4; int dof_id6 = 6*dof_id;
            int jid6 = dof_id6;
            // branch to get pointer locations
            int dstOffset; const T * src;
                 if (selector == 0){ dstOffset = 1512; src = grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, 1260); }
            else if (selector == 1){ dstOffset = 1554; src = grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, 1302); }
            else if (selector == 2){ dstOffset = 1596; src = &s_vaf[0]; }
            else              { dstOffset = 1638; src = &s_vaf[84]; }
            mx2_scaled<T>(grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, dstOffset + dof_id6), &src[jid6], 1);
        }
        __syncthreads();
        //
        // Forward Pass
        //
        // We start with dv/du noting that we only have values
        //    for ancestors and for the current index else 0
        // dv/du where bfs_level is 0
        //     joints are: A1
        //     links are: L1
        // when parent is base dv_dq = 0, dv_dqd = S
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int dq_flag = (ind / 6) == 0;
            int du_offset = dq_flag ? 0 : 168;
            (*grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, du_offset + 6*0 + row)) = (!dq_flag && row == 2) * static_cast<T>(1);
        }
        __syncthreads();
        // dv/du where bfs_level is 1
        //     joints are: A2
        //     links are: L2
        // dv/du = Xmat*dv_parent/du + {Mx(Xv) or S for col ind}
        // first compute dv/du = Xmat*dv_parent/du
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 1; int col_jid = col_du % 1;
            int dq_flag = col < 1;
            int du_col_offset = dq_flag * 0 + !dq_flag * 168 + 6 * col_jid;
            (*grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, du_col_offset + 6*1 + row)) = 
                dot_prod<T,6,6,1>(&s_XImats[36*1 + row],grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, du_col_offset + 6*0));
            // then add {Mx(Xv) or S for col ind}
            (*grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, du_col_offset + 6*1 + 6 + row)) = 
                dq_flag * (*grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, 1512 + 6*1 + row)) + (!dq_flag && row == 2) * static_cast<T>(1);
        }
        __syncthreads();
        // dv/du where bfs_level is 2
        //     joints are: A3
        //     links are: L3
        // dv/du = Xmat*dv_parent/du + {Mx(Xv) or S for col ind}
        // first compute dv/du = Xmat*dv_parent/du
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 24; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 2; int col_jid = col_du % 2;
            int dq_flag = col == col_du;
            int du_col_offset = dq_flag * 0 + !dq_flag * 168 + 6 * col_jid;
            (*grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, du_col_offset + 6*3 + row)) = 
                dot_prod<T,6,6,1>(&s_XImats[36*2 + row],grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, du_col_offset + 6*1));
            // then add {Mx(Xv) or S for col ind}
            if (col_jid == 1) {
                (*grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, du_col_offset + 6*3 + 6 + row)) = 
                    dq_flag * (*grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, 1512 + 6*2 + row)) + (!dq_flag && row == 2) * static_cast<T>(1);
            }
        }
        __syncthreads();
        // dv/du where bfs_level is 3
        //     joints are: A4
        //     links are: L4
        // dv/du = Xmat*dv_parent/du + {Mx(Xv) or S for col ind}
        // first compute dv/du = Xmat*dv_parent/du
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 3; int col_jid = col_du % 3;
            int dq_flag = col == col_du;
            int du_col_offset = dq_flag * 0 + !dq_flag * 168 + 6 * col_jid;
            (*grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, du_col_offset + 6*6 + row)) = 
                dot_prod<T,6,6,1>(&s_XImats[36*3 + row],grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, du_col_offset + 6*3));
            // then add {Mx(Xv) or S for col ind}
            if (col_jid == 2) {
                (*grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, du_col_offset + 6*6 + 6 + row)) = 
                    dq_flag * (*grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, 1512 + 6*3 + row)) + (!dq_flag && row == 2) * static_cast<T>(1);
            }
        }
        __syncthreads();
        // dv/du where bfs_level is 4
        //     joints are: A5
        //     links are: L5
        // dv/du = Xmat*dv_parent/du + {Mx(Xv) or S for col ind}
        // first compute dv/du = Xmat*dv_parent/du
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 48; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 4; int col_jid = col_du % 4;
            int dq_flag = col == col_du;
            int du_col_offset = dq_flag * 0 + !dq_flag * 168 + 6 * col_jid;
            (*grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, du_col_offset + 6*10 + row)) = 
                dot_prod<T,6,6,1>(&s_XImats[36*4 + row],grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, du_col_offset + 6*6));
            // then add {Mx(Xv) or S for col ind}
            if (col_jid == 3) {
                (*grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, du_col_offset + 6*10 + 6 + row)) = 
                    dq_flag * (*grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, 1512 + 6*4 + row)) + (!dq_flag && row == 2) * static_cast<T>(1);
            }
        }
        __syncthreads();
        // dv/du where bfs_level is 5
        //     joints are: A6
        //     links are: L6
        // dv/du = Xmat*dv_parent/du + {Mx(Xv) or S for col ind}
        // first compute dv/du = Xmat*dv_parent/du
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 60; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 5; int col_jid = col_du % 5;
            int dq_flag = col == col_du;
            int du_col_offset = dq_flag * 0 + !dq_flag * 168 + 6 * col_jid;
            (*grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, du_col_offset + 6*15 + row)) = 
                dot_prod<T,6,6,1>(&s_XImats[36*5 + row],grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, du_col_offset + 6*10));
            // then add {Mx(Xv) or S for col ind}
            if (col_jid == 4) {
                (*grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, du_col_offset + 6*15 + 6 + row)) = 
                    dq_flag * (*grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, 1512 + 6*5 + row)) + (!dq_flag && row == 2) * static_cast<T>(1);
            }
        }
        __syncthreads();
        // dv/du where bfs_level is 6
        //     joints are: A7
        //     links are: L7
        // dv/du = Xmat*dv_parent/du + {Mx(Xv) or S for col ind}
        // first compute dv/du = Xmat*dv_parent/du
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 72; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 6; int col_jid = col_du % 6;
            int dq_flag = col == col_du;
            int du_col_offset = dq_flag * 0 + !dq_flag * 168 + 6 * col_jid;
            (*grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, du_col_offset + 6*21 + row)) = 
                dot_prod<T,6,6,1>(&s_XImats[36*6 + row],grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, du_col_offset + 6*15));
            // then add {Mx(Xv) or S for col ind}
            if (col_jid == 5) {
                (*grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, du_col_offset + 6*21 + 6 + row)) = 
                    dq_flag * (*grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, 1512 + 6*6 + row)) + (!dq_flag && row == 2) * static_cast<T>(1);
            }
        }
        __syncthreads();
        // start da/du by setting = MxS(dv/du)*qd + {MxXa, Mxv} for all n in parallel
        // start with da/du = MxS(dv/du)*qd
        for(int col = threadIdx.x + threadIdx.y*blockDim.x; col < 56; col += blockDim.x*blockDim.y){
            int col_du = col % 28;
            // non-branching pointer selector
            int jid = (col_du < 1) * 0 + (col_du < 3 && col_du >= 1) * 1 + (col_du < 6 && col_du >= 3) * 2 + (col_du < 10 && col_du >= 6) * 3 + (col_du < 15 && col_du >= 10) * 4 + (col_du < 21 && col_du >= 15) * 5 + (col_du >= 21) * 6;
            mx2_scaled<T>(grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, 336 + 6*col), grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, 0 + 6*col), (1) * s_qd[jid]);
            // then add {MxXa, Mxv} to the appropriate column
            int dq_flag = col == col_du; int src_offset = dq_flag * 1554 + !dq_flag * 1596 + 6*jid;
            if(col_du == ((jid+1)*(jid+2)/2 - 1)){
                for(int row = 0; row < 6; row++){
                    (*grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, 336 + 6*col + row)) += (*grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, src_offset + row));
                }
            }
        }
        __syncthreads();
        // Finish da/du with parent updates noting that we only have values
        //    for ancestors and for the current index and nothing for bfs 0
        // da/du where bfs_level is 1
        //     joints are: A2
        //     links are: L2
        // da/du += Xmat*da_parent/du
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 1;
            int dq_flag = col == col_du; int col_jid = col_du % 1;
            int du_col_offset = dq_flag * 336 + !dq_flag * 504 + 6 * col_jid;
            (*grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, du_col_offset + 6*1 + row)) += 
                dot_prod<T,6,6,1>(&s_XImats[36*1 + row],grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, du_col_offset + 6*0));
        }
        __syncthreads();
        // da/du where bfs_level is 2
        //     joints are: A3
        //     links are: L3
        // da/du += Xmat*da_parent/du
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 24; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 2;
            int dq_flag = col == col_du; int col_jid = col_du % 2;
            int du_col_offset = dq_flag * 336 + !dq_flag * 504 + 6 * col_jid;
            (*grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, du_col_offset + 6*3 + row)) += 
                dot_prod<T,6,6,1>(&s_XImats[36*2 + row],grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, du_col_offset + 6*1));
        }
        __syncthreads();
        // da/du where bfs_level is 3
        //     joints are: A4
        //     links are: L4
        // da/du += Xmat*da_parent/du
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 3;
            int dq_flag = col == col_du; int col_jid = col_du % 3;
            int du_col_offset = dq_flag * 336 + !dq_flag * 504 + 6 * col_jid;
            (*grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, du_col_offset + 6*6 + row)) += 
                dot_prod<T,6,6,1>(&s_XImats[36*3 + row],grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, du_col_offset + 6*3));
        }
        __syncthreads();
        // da/du where bfs_level is 4
        //     joints are: A5
        //     links are: L5
        // da/du += Xmat*da_parent/du
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 48; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 4;
            int dq_flag = col == col_du; int col_jid = col_du % 4;
            int du_col_offset = dq_flag * 336 + !dq_flag * 504 + 6 * col_jid;
            (*grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, du_col_offset + 6*10 + row)) += 
                dot_prod<T,6,6,1>(&s_XImats[36*4 + row],grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, du_col_offset + 6*6));
        }
        __syncthreads();
        // da/du where bfs_level is 5
        //     joints are: A6
        //     links are: L6
        // da/du += Xmat*da_parent/du
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 60; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 5;
            int dq_flag = col == col_du; int col_jid = col_du % 5;
            int du_col_offset = dq_flag * 336 + !dq_flag * 504 + 6 * col_jid;
            (*grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, du_col_offset + 6*15 + row)) += 
                dot_prod<T,6,6,1>(&s_XImats[36*5 + row],grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, du_col_offset + 6*10));
        }
        __syncthreads();
        // da/du where bfs_level is 6
        //     joints are: A7
        //     links are: L7
        // da/du += Xmat*da_parent/du
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 72; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 6;
            int dq_flag = col == col_du; int col_jid = col_du % 6;
            int du_col_offset = dq_flag * 336 + !dq_flag * 504 + 6 * col_jid;
            (*grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, du_col_offset + 6*21 + row)) += 
                dot_prod<T,6,6,1>(&s_XImats[36*6 + row],grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, du_col_offset + 6*15));
        }
        __syncthreads();
        // Init df/du to 0
        glass::set_const<T, 588>(static_cast<T>(0), grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, 672));
        // Start the df/du by setting = fx(dv/du)*Iv and also compute the temp = Fx(v)*I 
        //    aka do all of the Fx comps in parallel
        // note that while df has more cols than dva the dva cols are the first few df cols
        for(int col = threadIdx.x + threadIdx.y*blockDim.x; col < 98; col += blockDim.x*blockDim.y){
            int col_du = col % 28;
            // non-branching pointer selector
            int jid = (col_du < 1) * 0 + (col_du < 3 && col_du >= 1) * 1 + (col_du < 6 && col_du >= 3) * 2 + (col_du < 10 && col_du >= 6) * 3 + (col_du < 15 && col_du >= 10) * 4 + (col_du < 21 && col_du >= 15) * 5 + (col_du >= 21) * 6;
            // Compute Offsets and Pointers
            int dq_flag = col == col_du; int dva_to_df_adjust = 7*jid - jid*(jid+1)/2;
            int Offset_col_du_src = dq_flag * 0 + !dq_flag * 168 + 6*col_du;
            int Offset_col_du_dst = dq_flag * 672 + !dq_flag * 966 + 6*(col_du + dva_to_df_adjust);
            T *dst = grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, Offset_col_du_dst); const T *fx_src = grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, Offset_col_du_src); const T *mult_src = grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, 1680 + 6*jid);
            // Adjust pointers for temp comps (if applicable)
            if (col >= 56) {
                int comp = col - 56; int comp_col = comp % 6; // int jid = comp / 6;
                int jid6 = comp - comp_col; int jid36_col6 = 6*jid6 + 6*comp_col;
                dst = grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, 1260 + jid36_col6); fx_src = &s_vaf[jid6]; mult_src = &s_XImats[252 + jid36_col6];
            }
            fx_times_v<T>(dst, fx_src, mult_src);
        }
        __syncthreads();
        // Then in parallel finish df/du += I*da/du + (Fx(v)I)*dv/du
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 336; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col6 = ind - row; int col_du = (col % 28);
            // non-branching pointer selector
            int jid = (col_du < 1) * 0 + (col_du < 3 && col_du >= 1) * 1 + (col_du < 6 && col_du >= 3) * 2 + (col_du < 10 && col_du >= 6) * 3 + (col_du < 15 && col_du >= 10) * 4 + (col_du < 21 && col_du >= 15) * 5 + (col_du >= 21) * 6;
            // Compute Offsets and Pointers
            int dva_to_df_adjust = 7*jid - jid*(jid+1)/2;
            if (col >= 28){dva_to_df_adjust += 21;}
            T *df_row_col = grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, 672 + 6*dva_to_df_adjust + ind);
            const T *dv_col = grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, 0 + col6); const T *da_col = grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, 336 + col6);
            int jid36 = 36*jid; const T *I_row = &s_XImats[252 + jid36 + row]; const T *FxvI_row = grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, 1260 + jid36 + row);
            // Compute the values
            *df_row_col += dot_prod<T,6,6,1>(I_row,da_col) + dot_prod<T,6,6,1>(FxvI_row,dv_col);
        }
        // At the same time compute the last temp var: -X^T * mx(f)
        // use Mx(Xv) temp memory as those values are no longer needed
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 42; ind += blockDim.x*blockDim.y){
            int XTcol = ind % 6; int jid6 = ind - XTcol;
            (*grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, 1512 + ind)) = -dot_prod<T,6,1,1>(&s_XImats[6*(jid6 + XTcol)], grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, 1638 + jid6));
        }
        __syncthreads();
        //
        // BACKWARD Pass
        //
        // df/du update where bfs_level is 6
        //     joints are: A7
        //     links are: L7
        // df_lambda/du += X^T * df/du + {Xmx(f), 0}
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 84; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 7;
            int dq_flag = col == col_du;
            int du_col_offset = dq_flag * 672 + !dq_flag * 966 + 6*col_du;
            int dst_adjust = (col_du >= 6) * 6 * 0; // adjust for sparsity compression offsets
            T *dst = grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, du_col_offset + 6*35 + dst_adjust + row);
            T update_val = dot_prod<T,6,1,1>(&s_XImats[36*6 + 6*row],grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, du_col_offset + 6*42))
                          + dq_flag * (col_du == 6) * (*grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, 1512 + 6*6 + row));
            *dst += update_val;
        }
        __syncthreads();
        // df/du update where bfs_level is 5
        //     joints are: A6
        //     links are: L6
        // df_lambda/du += X^T * df/du + {Xmx(f), 0}
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 84; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 7;
            int dq_flag = col == col_du;
            int du_col_offset = dq_flag * 672 + !dq_flag * 966 + 6*col_du;
            int dst_adjust = (col_du >= 5) * 6 * 0; // adjust for sparsity compression offsets
            T *dst = grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, du_col_offset + 6*28 + dst_adjust + row);
            T update_val = dot_prod<T,6,1,1>(&s_XImats[36*5 + 6*row],grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, du_col_offset + 6*35))
                          + dq_flag * (col_du == 5) * (*grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, 1512 + 6*5 + row));
            *dst += update_val;
        }
        __syncthreads();
        // df/du update where bfs_level is 4
        //     joints are: A5
        //     links are: L5
        // df_lambda/du += X^T * df/du + {Xmx(f), 0}
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 84; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 7;
            int dq_flag = col == col_du;
            int du_col_offset = dq_flag * 672 + !dq_flag * 966 + 6*col_du;
            int dst_adjust = (col_du >= 4) * 6 * 0; // adjust for sparsity compression offsets
            T *dst = grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, du_col_offset + 6*21 + dst_adjust + row);
            T update_val = dot_prod<T,6,1,1>(&s_XImats[36*4 + 6*row],grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, du_col_offset + 6*28))
                          + dq_flag * (col_du == 4) * (*grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, 1512 + 6*4 + row));
            *dst += update_val;
        }
        __syncthreads();
        // df/du update where bfs_level is 3
        //     joints are: A4
        //     links are: L4
        // df_lambda/du += X^T * df/du + {Xmx(f), 0}
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 84; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 7;
            int dq_flag = col == col_du;
            int du_col_offset = dq_flag * 672 + !dq_flag * 966 + 6*col_du;
            int dst_adjust = (col_du >= 3) * 6 * 0; // adjust for sparsity compression offsets
            T *dst = grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, du_col_offset + 6*14 + dst_adjust + row);
            T update_val = dot_prod<T,6,1,1>(&s_XImats[36*3 + 6*row],grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, du_col_offset + 6*21))
                          + dq_flag * (col_du == 3) * (*grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, 1512 + 6*3 + row));
            *dst += update_val;
        }
        __syncthreads();
        // df/du update where bfs_level is 2
        //     joints are: A3
        //     links are: L3
        // df_lambda/du += X^T * df/du + {Xmx(f), 0}
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 84; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 7;
            int dq_flag = col == col_du;
            int du_col_offset = dq_flag * 672 + !dq_flag * 966 + 6*col_du;
            int dst_adjust = (col_du >= 2) * 6 * 0; // adjust for sparsity compression offsets
            T *dst = grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, du_col_offset + 6*7 + dst_adjust + row);
            T update_val = dot_prod<T,6,1,1>(&s_XImats[36*2 + 6*row],grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, du_col_offset + 6*14))
                          + dq_flag * (col_du == 2) * (*grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, 1512 + 6*2 + row));
            *dst += update_val;
        }
        __syncthreads();
        // df/du update where bfs_level is 1
        //     joints are: A2
        //     links are: L2
        // df_lambda/du += X^T * df/du + {Xmx(f), 0}
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 84; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 7;
            int dq_flag = col == col_du;
            int du_col_offset = dq_flag * 672 + !dq_flag * 966 + 6*col_du;
            int dst_adjust = (col_du >= 1) * 6 * 0; // adjust for sparsity compression offsets
            T *dst = grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, du_col_offset + 6*0 + dst_adjust + row);
            T update_val = dot_prod<T,6,1,1>(&s_XImats[36*1 + 6*row],grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, du_col_offset + 6*7))
                          + dq_flag * (col_du == 1) * (*grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, 1512 + 6*1 + row));
            *dst += update_val;
        }
        __syncthreads();
        // Finally dc[i]/du = S[i]^T*df[i]/du
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
            int jid = ind % 7; int jid_dq_qd = ind / 7; int jid_du = jid_dq_qd % 7; int dq_flag = jid_du == jid_dq_qd;
            int Offset_src = dq_flag * 672 + !dq_flag * 966 + 6 * 7 * jid + 6 * jid_du + 2;
            int Offset_dst = !dq_flag * 49 + 7 * jid_du + jid;
            s_dc_du[Offset_dst] = (1) * (*grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, Offset_src));
        }
        __syncthreads();
    }

    /**
     * inverse_dynamics_gradient orchestration as a single inner-owns-placement device function
     *
     * Notes:
     *   Owns the s_temp pool placement; the repoint covers every consumer below (incl. the XImats helper's sincos scratch)
     *
     * @param s_dc_du is the output buffer (caller places); size 2*NUM_VEL*NUM_VEL = 98
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param s_vaf is the id intermediate band (caller places); size 18*NUM_JOINTS = 126
     * @param s_qdd is the vector of joint accelerations
     * @param s_temp is the shared scratch pool (used when SCRATCH_IN_SMEM)
     * @param d_workspace is the global scratch pool (used when !SCRATCH_IN_SMEM)
     * @param d_temp_spill is the inverse_dynamics_gradient da_df band spill region (used when USE_DA_DF_SPILL)
     * @param s_XImats is the (shared) memory holding the updated XI matricies for the given s_q
     * @param s_topology_helpers is the (shared) memory location for the topology_helpers (nullptr/unused for serial chains with identical Ss)
     * @param d_robotModel holds XImats/topology; gravity is the gravity constant
     * @param d_f_ext is the (optional) GLOBAL external forces, body-major 6*NUM_BODIES local-frame, or nullptr
     */
    template <typename T, bool SCRATCH_IN_SMEM = true, bool USE_DA_DF_SPILL = false>
    __device__ __forceinline__
    void inverse_dynamics_gradient_device_qdd(T *s_dc_du, const T *s_q, const T *s_qd, T *s_vaf, const T *s_qdd, T *s_XImats, int *s_topology_helpers, T *s_temp, T *d_workspace, T *d_temp_spill, const robotModel<T> *d_robotModel, T *d_f_ext, const T gravity) {
        if constexpr (!SCRATCH_IN_SMEM) { s_temp = d_workspace; } else { (void)d_workspace; }
        load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
        inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_qdd, s_XImats, s_topology_helpers, s_temp, d_f_ext, gravity);
        inverse_dynamics_gradient_inner<T, USE_DA_DF_SPILL>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_topology_helpers, s_temp, d_temp_spill, gravity);
    }

    /**
     * inverse_dynamics_gradient orchestration as a single inner-owns-placement device function
     *
     * Notes:
     *   Owns the s_temp pool placement; the repoint covers every consumer below (incl. the XImats helper's sincos scratch)
     *
     * @param s_dc_du is the output buffer (caller places); size 2*NUM_VEL*NUM_VEL = 98
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param s_vaf is the id intermediate band (caller places); size 18*NUM_JOINTS = 126
     * @param s_temp is the shared scratch pool (used when SCRATCH_IN_SMEM)
     * @param d_workspace is the global scratch pool (used when !SCRATCH_IN_SMEM)
     * @param d_temp_spill is the inverse_dynamics_gradient da_df band spill region (used when USE_DA_DF_SPILL)
     * @param s_XImats is the (shared) memory holding the updated XI matricies for the given s_q
     * @param s_topology_helpers is the (shared) memory location for the topology_helpers (nullptr/unused for serial chains with identical Ss)
     * @param d_robotModel holds XImats/topology; gravity is the gravity constant
     * @param d_f_ext is the (optional) GLOBAL external forces, body-major 6*NUM_BODIES local-frame, or nullptr
     */
    template <typename T, bool SCRATCH_IN_SMEM = true, bool USE_DA_DF_SPILL = false>
    __device__ __forceinline__
    void inverse_dynamics_gradient_device(T *s_dc_du, const T *s_q, const T *s_qd, T *s_vaf, T *s_XImats, int *s_topology_helpers, T *s_temp, T *d_workspace, T *d_temp_spill, const robotModel<T> *d_robotModel, T *d_f_ext, const T gravity) {
        if constexpr (!SCRATCH_IN_SMEM) { s_temp = d_workspace; } else { (void)d_workspace; }
        load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
        inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_XImats, s_topology_helpers, s_temp, d_f_ext, gravity);
        inverse_dynamics_gradient_inner<T, USE_DA_DF_SPILL>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_topology_helpers, s_temp, d_temp_spill, gravity);
    }

    /**
     * Computes the gradient of inverse dynamics
     *
     * @param d_dc_du is a pointer to memory for the final result of size 2*NUM_VEL*NUM_VEL = 98
     * @param d_q_dq is the vector of joint positions and velocities
     * @param stride_q_qd is the stide between each q, qd
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param d_f_ext is the (optional) GLOBAL external forces, body-major 6*NUM_BODIES local-frame, or nullptr
     * @param d_qdd is the vector of joint accelerations
     * @param gravity is the gravity constant
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER, bool MUJOCO_OUTPUT = false>
    __global__
    __launch_bounds__(tier_max_threads<RESOURCE_TIER>())
    void inverse_dynamics_gradient_kernel_single_timing(T *d_dc_du, unsigned char *d_workspace, const T *d_q_qd, const int stride_q_qd, const T *d_qdd, T *d_f_ext, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        if constexpr (RESOURCE_TIER == TIER_SHARED) {
            // GRID shared arena layout
            //   T s_q_qd[14]
            //   T s_dc_du[98]
            //   T s_vaf[126]
            //   T s_qdd[7]
            //   T s_XImats[504]
            //   T s_temp[1722]
            //   bytes s_linalg_smem[GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()]
            extern __shared__ __align__(16) unsigned char s_arena[];
            size_t s_arena_offset = 0;
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_q_qd = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(14);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_dc_du = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(98);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_vaf = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(126);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_qdd = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(7);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_XImats = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(504);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_temp = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(1722);
            int *s_topology_helpers = nullptr;
            unsigned char *s_linalg_smem = nullptr;
            if (static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) > 0) {
                s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
                s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
                s_arena_offset += static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());
            }
            #ifdef GRID_CUDA_DEBUG_LAYOUT
            assert(s_arena_offset == grid_shared_arena_bytes<T>(2471, 0, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()));
            #endif
            (void)s_arena_offset;
            T *d_temp_spill = nullptr; (void)d_temp_spill;
            T *s_q = s_q_qd; T *s_qd = &s_q_qd[7];
            // load to shared mem
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
                s_q_qd[ind] = d_q_qd[ind];
            }
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                s_qdd[ind] = d_qdd[ind];
            }
            __syncthreads();
            // compute with NUM_TIMESTEPS as NUM_REPS for timing
            for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
                // anti-LICM: volatile reload of inputs each rep
                for(int _aopt_i = threadIdx.x + threadIdx.y*blockDim.x; _aopt_i < 14; _aopt_i += blockDim.x*blockDim.y){
                    reinterpret_cast<volatile T *>(s_q_qd)[_aopt_i] = reinterpret_cast<const volatile T *>(d_q_qd)[_aopt_i];
                }
                for(int _aopt_i = threadIdx.x + threadIdx.y*blockDim.x; _aopt_i < 7; _aopt_i += blockDim.x*blockDim.y){
                    reinterpret_cast<volatile T *>(s_qdd)[_aopt_i] = reinterpret_cast<const volatile T *>(d_qdd)[_aopt_i];
                }
                __syncthreads();
                // anti-LICM (1/2): stomp one input slot with `rep`
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
                    reinterpret_cast<volatile T *>(s_q_qd)[rep % (14)] = static_cast<T>(rep);
                    reinterpret_cast<volatile T *>(s_qdd)[rep % (7)] = static_cast<T>(rep);
                }
                // anti-LICM (2/2): feedback prev rep's d_dc_du into s_q_qd (true loop-carried dep)
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
                    T _aopt_fb1 = reinterpret_cast<const volatile T *>(d_dc_du)[(rep + 0x3FF) & 0x3FF];
                    T _aopt_fb2 = reinterpret_cast<const volatile T *>(d_dc_du)[(rep + 0x3FE) & 0x3FF];
                    T _aopt_fb3 = reinterpret_cast<const volatile T *>(d_dc_du)[(rep + 0x3FD) & 0x3FF];
                    reinterpret_cast<volatile T *>(s_q_qd)[(rep + 1) % (14)] += _aopt_fb1;
                    reinterpret_cast<volatile T *>(s_q_qd)[(rep + 2) % (14)] += _aopt_fb2;
                    reinterpret_cast<volatile T *>(s_q_qd)[(rep + 3) % (14)] += _aopt_fb3;
                }
                __syncthreads();
                inverse_dynamics_gradient_device_qdd<T, true, false>(s_dc_du, s_q, s_qd, s_vaf, s_qdd, s_XImats, s_topology_helpers, s_temp, nullptr, nullptr, d_robotModel, d_f_ext, gravity);
                __syncthreads();
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) { reinterpret_cast<volatile T *>(d_dc_du)[rep & 1023] = reinterpret_cast<const volatile T *>(s_dc_du)[rep & 7]; }
            }
            // save down to global
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
                d_dc_du[ind] = s_dc_du[ind];
            }
            __syncthreads();
        }
        else if constexpr (RESOURCE_TIER == TIER_LITE) {
            // GRID shared arena layout
            //   T s_q_qd[14]
            //   T s_dc_du[98]
            //   T s_vaf[126]
            //   T s_qdd[7]
            //   T s_XImats[504]
            //   T s_temp[1722]
            //   bytes s_linalg_smem[GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()]
            extern __shared__ __align__(16) unsigned char s_arena[];
            size_t s_arena_offset = 0;
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_q_qd = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(14);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_dc_du = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(98);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_vaf = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(126);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_qdd = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(7);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_XImats = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(504);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_temp = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(1722);
            int *s_topology_helpers = nullptr;
            unsigned char *s_linalg_smem = nullptr;
            if (static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) > 0) {
                s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
                s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
                s_arena_offset += static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());
            }
            #ifdef GRID_CUDA_DEBUG_LAYOUT
            assert(s_arena_offset == grid_shared_arena_bytes<T>(2471, 0, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()));
            #endif
            (void)s_arena_offset;
            T *d_temp_spill = nullptr; (void)d_temp_spill;
            T *s_q = s_q_qd; T *s_qd = &s_q_qd[7];
            // load to shared mem
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
                s_q_qd[ind] = d_q_qd[ind];
            }
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                s_qdd[ind] = d_qdd[ind];
            }
            __syncthreads();
            // compute with NUM_TIMESTEPS as NUM_REPS for timing
            for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
                // anti-LICM: volatile reload of inputs each rep
                for(int _aopt_i = threadIdx.x + threadIdx.y*blockDim.x; _aopt_i < 14; _aopt_i += blockDim.x*blockDim.y){
                    reinterpret_cast<volatile T *>(s_q_qd)[_aopt_i] = reinterpret_cast<const volatile T *>(d_q_qd)[_aopt_i];
                }
                for(int _aopt_i = threadIdx.x + threadIdx.y*blockDim.x; _aopt_i < 7; _aopt_i += blockDim.x*blockDim.y){
                    reinterpret_cast<volatile T *>(s_qdd)[_aopt_i] = reinterpret_cast<const volatile T *>(d_qdd)[_aopt_i];
                }
                __syncthreads();
                // anti-LICM (1/2): stomp one input slot with `rep`
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
                    reinterpret_cast<volatile T *>(s_q_qd)[rep % (14)] = static_cast<T>(rep);
                    reinterpret_cast<volatile T *>(s_qdd)[rep % (7)] = static_cast<T>(rep);
                }
                // anti-LICM (2/2): feedback prev rep's d_dc_du into s_q_qd (true loop-carried dep)
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
                    T _aopt_fb1 = reinterpret_cast<const volatile T *>(d_dc_du)[(rep + 0x3FF) & 0x3FF];
                    T _aopt_fb2 = reinterpret_cast<const volatile T *>(d_dc_du)[(rep + 0x3FE) & 0x3FF];
                    T _aopt_fb3 = reinterpret_cast<const volatile T *>(d_dc_du)[(rep + 0x3FD) & 0x3FF];
                    reinterpret_cast<volatile T *>(s_q_qd)[(rep + 1) % (14)] += _aopt_fb1;
                    reinterpret_cast<volatile T *>(s_q_qd)[(rep + 2) % (14)] += _aopt_fb2;
                    reinterpret_cast<volatile T *>(s_q_qd)[(rep + 3) % (14)] += _aopt_fb3;
                }
                __syncthreads();
                inverse_dynamics_gradient_device_qdd<T, true, false>(s_dc_du, s_q, s_qd, s_vaf, s_qdd, s_XImats, s_topology_helpers, s_temp, nullptr, nullptr, d_robotModel, d_f_ext, gravity);
                __syncthreads();
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) { reinterpret_cast<volatile T *>(d_dc_du)[rep & 1023] = reinterpret_cast<const volatile T *>(s_dc_du)[rep & 7]; }
            }
            // save down to global
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
                d_dc_du[ind] = s_dc_du[ind];
            }
            __syncthreads();
        }
        else if constexpr (RESOURCE_TIER == TIER_MINIMAL) {
            // GRID shared arena layout
            //   T s_q_qd[14]
            //   T s_dc_du[98]
            //   T s_vaf[126]
            //   T s_qdd[7]
            //   T s_XImats[504]
            //   bytes s_linalg_smem[GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()]
            extern __shared__ __align__(16) unsigned char s_arena[];
            size_t s_arena_offset = 0;
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_q_qd = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(14);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_dc_du = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(98);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_vaf = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(126);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_qdd = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(7);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_XImats = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(504);
            T *s_temp = nullptr;
            int *s_topology_helpers = nullptr;
            unsigned char *s_linalg_smem = nullptr;
            if (static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) > 0) {
                s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
                s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
                s_arena_offset += static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());
            }
            #ifdef GRID_CUDA_DEBUG_LAYOUT
            assert(s_arena_offset == grid_shared_arena_bytes<T>(749, 0, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()));
            #endif
            (void)s_arena_offset;
            T *d_temp_spill = nullptr; (void)d_temp_spill;
            T *s_q = s_q_qd; T *s_qd = &s_q_qd[7];
            // load to shared mem
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
                s_q_qd[ind] = d_q_qd[ind];
            }
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                s_qdd[ind] = d_qdd[ind];
            }
            __syncthreads();
            // compute with NUM_TIMESTEPS as NUM_REPS for timing
            for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
                // anti-LICM: volatile reload of inputs each rep
                for(int _aopt_i = threadIdx.x + threadIdx.y*blockDim.x; _aopt_i < 14; _aopt_i += blockDim.x*blockDim.y){
                    reinterpret_cast<volatile T *>(s_q_qd)[_aopt_i] = reinterpret_cast<const volatile T *>(d_q_qd)[_aopt_i];
                }
                for(int _aopt_i = threadIdx.x + threadIdx.y*blockDim.x; _aopt_i < 7; _aopt_i += blockDim.x*blockDim.y){
                    reinterpret_cast<volatile T *>(s_qdd)[_aopt_i] = reinterpret_cast<const volatile T *>(d_qdd)[_aopt_i];
                }
                __syncthreads();
                // anti-LICM (1/2): stomp one input slot with `rep`
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
                    reinterpret_cast<volatile T *>(s_q_qd)[rep % (14)] = static_cast<T>(rep);
                    reinterpret_cast<volatile T *>(s_qdd)[rep % (7)] = static_cast<T>(rep);
                }
                // anti-LICM (2/2): feedback prev rep's d_dc_du into s_q_qd (true loop-carried dep)
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
                    T _aopt_fb1 = reinterpret_cast<const volatile T *>(d_dc_du)[(rep + 0x3FF) & 0x3FF];
                    T _aopt_fb2 = reinterpret_cast<const volatile T *>(d_dc_du)[(rep + 0x3FE) & 0x3FF];
                    T _aopt_fb3 = reinterpret_cast<const volatile T *>(d_dc_du)[(rep + 0x3FD) & 0x3FF];
                    reinterpret_cast<volatile T *>(s_q_qd)[(rep + 1) % (14)] += _aopt_fb1;
                    reinterpret_cast<volatile T *>(s_q_qd)[(rep + 2) % (14)] += _aopt_fb2;
                    reinterpret_cast<volatile T *>(s_q_qd)[(rep + 3) % (14)] += _aopt_fb3;
                }
                __syncthreads();
                inverse_dynamics_gradient_device_qdd<T, false, false>(s_dc_du, s_q, s_qd, s_vaf, s_qdd, s_XImats, s_topology_helpers, s_temp, reinterpret_cast<T *>(d_workspace), nullptr, d_robotModel, d_f_ext, gravity);
                __syncthreads();
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) { reinterpret_cast<volatile T *>(d_dc_du)[rep & 1023] = reinterpret_cast<const volatile T *>(s_dc_du)[rep & 7]; }
            }
            // save down to global
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
                d_dc_du[ind] = s_dc_du[ind];
            }
            __syncthreads();
        }
    }

    /**
     * Computes the gradient of inverse dynamics
     *
     * @param d_dc_du is a pointer to memory for the final result of size 2*NUM_VEL*NUM_VEL = 98
     * @param d_q_dq is the vector of joint positions and velocities
     * @param stride_q_qd is the stide between each q, qd
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param d_f_ext is the (optional) GLOBAL external forces, body-major 6*NUM_BODIES local-frame, or nullptr
     * @param d_qdd is the vector of joint accelerations
     * @param gravity is the gravity constant
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER, bool MUJOCO_OUTPUT = false>
    __global__
    __launch_bounds__(tier_max_threads<RESOURCE_TIER>())
    void inverse_dynamics_gradient_kernel(T *d_dc_du, unsigned char *d_workspace, const T *d_q_qd, const int stride_q_qd, const T *d_qdd, T *d_f_ext, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        if constexpr (RESOURCE_TIER == TIER_SHARED) {
            // GRID shared arena layout
            //   T s_q_qd[14]
            //   T s_dc_du[98]
            //   T s_vaf[126]
            //   T s_qdd[7]
            //   T s_XImats[504]
            //   T s_temp[1722]
            //   bytes s_linalg_smem[GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()]
            extern __shared__ __align__(16) unsigned char s_arena[];
            size_t s_arena_offset = 0;
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_q_qd = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(14);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_dc_du = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(98);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_vaf = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(126);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_qdd = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(7);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_XImats = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(504);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_temp = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(1722);
            int *s_topology_helpers = nullptr;
            unsigned char *s_linalg_smem = nullptr;
            if (static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) > 0) {
                s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
                s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
                s_arena_offset += static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());
            }
            #ifdef GRID_CUDA_DEBUG_LAYOUT
            assert(s_arena_offset == grid_shared_arena_bytes<T>(2471, 0, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()));
            #endif
            (void)s_arena_offset;
            T *d_temp_spill = nullptr; (void)d_temp_spill;
            T *s_q = s_q_qd; T *s_qd = &s_q_qd[7];
            for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
                // load to shared mem
                const T *d_q_qd_k = &d_q_qd[k*stride_q_qd];
                for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
                    s_q_qd[ind] = d_q_qd_k[ind];
                }
                const T *d_qdd_k = &d_qdd[k*7];
                for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                    s_qdd[ind] = d_qdd_k[ind];
                }
                __syncthreads();
                // compute — the orchestration inner owns its s_temp pool placement
                inverse_dynamics_gradient_device_qdd<T, true, false>(s_dc_du, s_q, s_qd, s_vaf, s_qdd, s_XImats, s_topology_helpers, s_temp, nullptr, nullptr, d_robotModel, d_f_ext, gravity);
                __syncthreads();
                // save down to global
                T *d_dc_du_k = &d_dc_du[k*98];
                for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
                    d_dc_du_k[ind] = s_dc_du[ind];
                }
                __syncthreads();
            }
        }
        else if constexpr (RESOURCE_TIER == TIER_LITE) {
            // GRID shared arena layout
            //   T s_q_qd[14]
            //   T s_dc_du[98]
            //   T s_vaf[126]
            //   T s_qdd[7]
            //   T s_XImats[504]
            //   T s_temp[1722]
            //   bytes s_linalg_smem[GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()]
            extern __shared__ __align__(16) unsigned char s_arena[];
            size_t s_arena_offset = 0;
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_q_qd = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(14);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_dc_du = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(98);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_vaf = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(126);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_qdd = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(7);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_XImats = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(504);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_temp = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(1722);
            int *s_topology_helpers = nullptr;
            unsigned char *s_linalg_smem = nullptr;
            if (static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) > 0) {
                s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
                s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
                s_arena_offset += static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());
            }
            #ifdef GRID_CUDA_DEBUG_LAYOUT
            assert(s_arena_offset == grid_shared_arena_bytes<T>(2471, 0, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()));
            #endif
            (void)s_arena_offset;
            T *d_temp_spill = nullptr; (void)d_temp_spill;
            T *s_q = s_q_qd; T *s_qd = &s_q_qd[7];
            for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
                // load to shared mem
                const T *d_q_qd_k = &d_q_qd[k*stride_q_qd];
                for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
                    s_q_qd[ind] = d_q_qd_k[ind];
                }
                const T *d_qdd_k = &d_qdd[k*7];
                for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                    s_qdd[ind] = d_qdd_k[ind];
                }
                __syncthreads();
                // compute — the orchestration inner owns its s_temp pool placement
                inverse_dynamics_gradient_device_qdd<T, true, false>(s_dc_du, s_q, s_qd, s_vaf, s_qdd, s_XImats, s_topology_helpers, s_temp, nullptr, nullptr, d_robotModel, d_f_ext, gravity);
                __syncthreads();
                // save down to global
                T *d_dc_du_k = &d_dc_du[k*98];
                for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
                    d_dc_du_k[ind] = s_dc_du[ind];
                }
                __syncthreads();
            }
        }
        else if constexpr (RESOURCE_TIER == TIER_MINIMAL) {
            // GRID shared arena layout
            //   T s_q_qd[14]
            //   T s_dc_du[98]
            //   T s_vaf[126]
            //   T s_qdd[7]
            //   T s_XImats[504]
            //   bytes s_linalg_smem[GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()]
            extern __shared__ __align__(16) unsigned char s_arena[];
            size_t s_arena_offset = 0;
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_q_qd = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(14);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_dc_du = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(98);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_vaf = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(126);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_qdd = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(7);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_XImats = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(504);
            T *s_temp = nullptr;
            int *s_topology_helpers = nullptr;
            unsigned char *s_linalg_smem = nullptr;
            if (static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) > 0) {
                s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
                s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
                s_arena_offset += static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());
            }
            #ifdef GRID_CUDA_DEBUG_LAYOUT
            assert(s_arena_offset == grid_shared_arena_bytes<T>(749, 0, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()));
            #endif
            (void)s_arena_offset;
            T *d_temp_spill = nullptr; (void)d_temp_spill;
            T *s_q = s_q_qd; T *s_qd = &s_q_qd[7];
            for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
                // load to shared mem
                const T *d_q_qd_k = &d_q_qd[k*stride_q_qd];
                for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
                    s_q_qd[ind] = d_q_qd_k[ind];
                }
                const T *d_qdd_k = &d_qdd[k*7];
                for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                    s_qdd[ind] = d_qdd_k[ind];
                }
                __syncthreads();
                // compute — the orchestration inner owns its s_temp pool placement
                inverse_dynamics_gradient_device_qdd<T, false, false>(s_dc_du, s_q, s_qd, s_vaf, s_qdd, s_XImats, s_topology_helpers, s_temp, reinterpret_cast<T *>(&d_workspace[grid_workspace_slot()*GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()]), nullptr, d_robotModel, d_f_ext, gravity);
                __syncthreads();
                // save down to global
                T *d_dc_du_k = &d_dc_du[k*98];
                for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
                    d_dc_du_k[ind] = s_dc_du[ind];
                }
                __syncthreads();
            }
        }
    }

    /**
     * Computes the gradient of inverse dynamics
     *
     * Notes:
     *   optimized for qdd = 0
     *
     * @param d_dc_du is a pointer to memory for the final result of size 2*NUM_VEL*NUM_VEL = 98
     * @param d_q_dq is the vector of joint positions and velocities
     * @param stride_q_qd is the stide between each q, qd
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param d_f_ext is the (optional) GLOBAL external forces, body-major 6*NUM_BODIES local-frame, or nullptr
     * @param gravity is the gravity constant
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER, bool MUJOCO_OUTPUT = false>
    __global__
    __launch_bounds__(tier_max_threads<RESOURCE_TIER>())
    void inverse_dynamics_gradient_kernel_single_timing(T *d_dc_du, unsigned char *d_workspace, const T *d_q_qd, const int stride_q_qd, T *d_f_ext, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        if constexpr (RESOURCE_TIER == TIER_SHARED) {
            // GRID shared arena layout
            //   T s_q_qd[14]
            //   T s_dc_du[98]
            //   T s_vaf[126]
            //   T s_XImats[504]
            //   T s_temp[1722]
            //   bytes s_linalg_smem[GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()]
            extern __shared__ __align__(16) unsigned char s_arena[];
            size_t s_arena_offset = 0;
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_q_qd = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(14);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_dc_du = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(98);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_vaf = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(126);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_XImats = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(504);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_temp = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(1722);
            int *s_topology_helpers = nullptr;
            unsigned char *s_linalg_smem = nullptr;
            if (static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) > 0) {
                s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
                s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
                s_arena_offset += static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());
            }
            #ifdef GRID_CUDA_DEBUG_LAYOUT
            assert(s_arena_offset == grid_shared_arena_bytes<T>(2464, 0, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()));
            #endif
            (void)s_arena_offset;
            T *d_temp_spill = nullptr; (void)d_temp_spill;
            T *s_q = s_q_qd; T *s_qd = &s_q_qd[7];
            // load to shared mem
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
                s_q_qd[ind] = d_q_qd[ind];
            }
            __syncthreads();
            // compute with NUM_TIMESTEPS as NUM_REPS for timing
            for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
                // anti-LICM: volatile reload of inputs each rep
                for(int _aopt_i = threadIdx.x + threadIdx.y*blockDim.x; _aopt_i < 14; _aopt_i += blockDim.x*blockDim.y){
                    reinterpret_cast<volatile T *>(s_q_qd)[_aopt_i] = reinterpret_cast<const volatile T *>(d_q_qd)[_aopt_i];
                }
                __syncthreads();
                // anti-LICM (1/2): stomp one input slot with `rep`
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
                    reinterpret_cast<volatile T *>(s_q_qd)[rep % (14)] = static_cast<T>(rep);
                }
                // anti-LICM (2/2): feedback prev rep's d_dc_du into s_q_qd (true loop-carried dep)
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
                    T _aopt_fb1 = reinterpret_cast<const volatile T *>(d_dc_du)[(rep + 0x3FF) & 0x3FF];
                    T _aopt_fb2 = reinterpret_cast<const volatile T *>(d_dc_du)[(rep + 0x3FE) & 0x3FF];
                    T _aopt_fb3 = reinterpret_cast<const volatile T *>(d_dc_du)[(rep + 0x3FD) & 0x3FF];
                    reinterpret_cast<volatile T *>(s_q_qd)[(rep + 1) % (14)] += _aopt_fb1;
                    reinterpret_cast<volatile T *>(s_q_qd)[(rep + 2) % (14)] += _aopt_fb2;
                    reinterpret_cast<volatile T *>(s_q_qd)[(rep + 3) % (14)] += _aopt_fb3;
                }
                __syncthreads();
                inverse_dynamics_gradient_device<T, true, false>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_topology_helpers, s_temp, nullptr, nullptr, d_robotModel, d_f_ext, gravity);
                __syncthreads();
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) { reinterpret_cast<volatile T *>(d_dc_du)[rep & 1023] = reinterpret_cast<const volatile T *>(s_dc_du)[rep & 7]; }
            }
            // save down to global
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
                d_dc_du[ind] = s_dc_du[ind];
            }
            __syncthreads();
        }
        else if constexpr (RESOURCE_TIER == TIER_LITE) {
            // GRID shared arena layout
            //   T s_q_qd[14]
            //   T s_dc_du[98]
            //   T s_vaf[126]
            //   T s_XImats[504]
            //   T s_temp[1722]
            //   bytes s_linalg_smem[GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()]
            extern __shared__ __align__(16) unsigned char s_arena[];
            size_t s_arena_offset = 0;
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_q_qd = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(14);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_dc_du = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(98);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_vaf = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(126);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_XImats = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(504);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_temp = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(1722);
            int *s_topology_helpers = nullptr;
            unsigned char *s_linalg_smem = nullptr;
            if (static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) > 0) {
                s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
                s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
                s_arena_offset += static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());
            }
            #ifdef GRID_CUDA_DEBUG_LAYOUT
            assert(s_arena_offset == grid_shared_arena_bytes<T>(2464, 0, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()));
            #endif
            (void)s_arena_offset;
            T *d_temp_spill = nullptr; (void)d_temp_spill;
            T *s_q = s_q_qd; T *s_qd = &s_q_qd[7];
            // load to shared mem
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
                s_q_qd[ind] = d_q_qd[ind];
            }
            __syncthreads();
            // compute with NUM_TIMESTEPS as NUM_REPS for timing
            for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
                // anti-LICM: volatile reload of inputs each rep
                for(int _aopt_i = threadIdx.x + threadIdx.y*blockDim.x; _aopt_i < 14; _aopt_i += blockDim.x*blockDim.y){
                    reinterpret_cast<volatile T *>(s_q_qd)[_aopt_i] = reinterpret_cast<const volatile T *>(d_q_qd)[_aopt_i];
                }
                __syncthreads();
                // anti-LICM (1/2): stomp one input slot with `rep`
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
                    reinterpret_cast<volatile T *>(s_q_qd)[rep % (14)] = static_cast<T>(rep);
                }
                // anti-LICM (2/2): feedback prev rep's d_dc_du into s_q_qd (true loop-carried dep)
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
                    T _aopt_fb1 = reinterpret_cast<const volatile T *>(d_dc_du)[(rep + 0x3FF) & 0x3FF];
                    T _aopt_fb2 = reinterpret_cast<const volatile T *>(d_dc_du)[(rep + 0x3FE) & 0x3FF];
                    T _aopt_fb3 = reinterpret_cast<const volatile T *>(d_dc_du)[(rep + 0x3FD) & 0x3FF];
                    reinterpret_cast<volatile T *>(s_q_qd)[(rep + 1) % (14)] += _aopt_fb1;
                    reinterpret_cast<volatile T *>(s_q_qd)[(rep + 2) % (14)] += _aopt_fb2;
                    reinterpret_cast<volatile T *>(s_q_qd)[(rep + 3) % (14)] += _aopt_fb3;
                }
                __syncthreads();
                inverse_dynamics_gradient_device<T, true, false>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_topology_helpers, s_temp, nullptr, nullptr, d_robotModel, d_f_ext, gravity);
                __syncthreads();
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) { reinterpret_cast<volatile T *>(d_dc_du)[rep & 1023] = reinterpret_cast<const volatile T *>(s_dc_du)[rep & 7]; }
            }
            // save down to global
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
                d_dc_du[ind] = s_dc_du[ind];
            }
            __syncthreads();
        }
        else if constexpr (RESOURCE_TIER == TIER_MINIMAL) {
            // GRID shared arena layout
            //   T s_q_qd[14]
            //   T s_dc_du[98]
            //   T s_vaf[126]
            //   T s_XImats[504]
            //   bytes s_linalg_smem[GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()]
            extern __shared__ __align__(16) unsigned char s_arena[];
            size_t s_arena_offset = 0;
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_q_qd = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(14);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_dc_du = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(98);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_vaf = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(126);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_XImats = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(504);
            T *s_temp = nullptr;
            int *s_topology_helpers = nullptr;
            unsigned char *s_linalg_smem = nullptr;
            if (static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) > 0) {
                s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
                s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
                s_arena_offset += static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());
            }
            #ifdef GRID_CUDA_DEBUG_LAYOUT
            assert(s_arena_offset == grid_shared_arena_bytes<T>(742, 0, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()));
            #endif
            (void)s_arena_offset;
            T *d_temp_spill = nullptr; (void)d_temp_spill;
            T *s_q = s_q_qd; T *s_qd = &s_q_qd[7];
            // load to shared mem
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
                s_q_qd[ind] = d_q_qd[ind];
            }
            __syncthreads();
            // compute with NUM_TIMESTEPS as NUM_REPS for timing
            for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
                // anti-LICM: volatile reload of inputs each rep
                for(int _aopt_i = threadIdx.x + threadIdx.y*blockDim.x; _aopt_i < 14; _aopt_i += blockDim.x*blockDim.y){
                    reinterpret_cast<volatile T *>(s_q_qd)[_aopt_i] = reinterpret_cast<const volatile T *>(d_q_qd)[_aopt_i];
                }
                __syncthreads();
                // anti-LICM (1/2): stomp one input slot with `rep`
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
                    reinterpret_cast<volatile T *>(s_q_qd)[rep % (14)] = static_cast<T>(rep);
                }
                // anti-LICM (2/2): feedback prev rep's d_dc_du into s_q_qd (true loop-carried dep)
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
                    T _aopt_fb1 = reinterpret_cast<const volatile T *>(d_dc_du)[(rep + 0x3FF) & 0x3FF];
                    T _aopt_fb2 = reinterpret_cast<const volatile T *>(d_dc_du)[(rep + 0x3FE) & 0x3FF];
                    T _aopt_fb3 = reinterpret_cast<const volatile T *>(d_dc_du)[(rep + 0x3FD) & 0x3FF];
                    reinterpret_cast<volatile T *>(s_q_qd)[(rep + 1) % (14)] += _aopt_fb1;
                    reinterpret_cast<volatile T *>(s_q_qd)[(rep + 2) % (14)] += _aopt_fb2;
                    reinterpret_cast<volatile T *>(s_q_qd)[(rep + 3) % (14)] += _aopt_fb3;
                }
                __syncthreads();
                inverse_dynamics_gradient_device<T, false, false>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_topology_helpers, s_temp, reinterpret_cast<T *>(d_workspace), nullptr, d_robotModel, d_f_ext, gravity);
                __syncthreads();
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) { reinterpret_cast<volatile T *>(d_dc_du)[rep & 1023] = reinterpret_cast<const volatile T *>(s_dc_du)[rep & 7]; }
            }
            // save down to global
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
                d_dc_du[ind] = s_dc_du[ind];
            }
            __syncthreads();
        }
    }

    /**
     * Computes the gradient of inverse dynamics
     *
     * Notes:
     *   optimized for qdd = 0
     *
     * @param d_dc_du is a pointer to memory for the final result of size 2*NUM_VEL*NUM_VEL = 98
     * @param d_q_dq is the vector of joint positions and velocities
     * @param stride_q_qd is the stide between each q, qd
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param d_f_ext is the (optional) GLOBAL external forces, body-major 6*NUM_BODIES local-frame, or nullptr
     * @param gravity is the gravity constant
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER, bool MUJOCO_OUTPUT = false>
    __global__
    __launch_bounds__(tier_max_threads<RESOURCE_TIER>())
    void inverse_dynamics_gradient_kernel(T *d_dc_du, unsigned char *d_workspace, const T *d_q_qd, const int stride_q_qd, T *d_f_ext, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        if constexpr (RESOURCE_TIER == TIER_SHARED) {
            // GRID shared arena layout
            //   T s_q_qd[14]
            //   T s_dc_du[98]
            //   T s_vaf[126]
            //   T s_XImats[504]
            //   T s_temp[1722]
            //   bytes s_linalg_smem[GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()]
            extern __shared__ __align__(16) unsigned char s_arena[];
            size_t s_arena_offset = 0;
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_q_qd = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(14);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_dc_du = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(98);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_vaf = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(126);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_XImats = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(504);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_temp = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(1722);
            int *s_topology_helpers = nullptr;
            unsigned char *s_linalg_smem = nullptr;
            if (static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) > 0) {
                s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
                s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
                s_arena_offset += static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());
            }
            #ifdef GRID_CUDA_DEBUG_LAYOUT
            assert(s_arena_offset == grid_shared_arena_bytes<T>(2464, 0, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()));
            #endif
            (void)s_arena_offset;
            T *d_temp_spill = nullptr; (void)d_temp_spill;
            T *s_q = s_q_qd; T *s_qd = &s_q_qd[7];
            for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
                // load to shared mem
                const T *d_q_qd_k = &d_q_qd[k*stride_q_qd];
                for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
                    s_q_qd[ind] = d_q_qd_k[ind];
                }
                __syncthreads();
                // compute — the orchestration inner owns its s_temp pool placement
                inverse_dynamics_gradient_device<T, true, false>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_topology_helpers, s_temp, nullptr, nullptr, d_robotModel, d_f_ext, gravity);
                __syncthreads();
                // save down to global
                T *d_dc_du_k = &d_dc_du[k*98];
                for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
                    d_dc_du_k[ind] = s_dc_du[ind];
                }
                __syncthreads();
            }
        }
        else if constexpr (RESOURCE_TIER == TIER_LITE) {
            // GRID shared arena layout
            //   T s_q_qd[14]
            //   T s_dc_du[98]
            //   T s_vaf[126]
            //   T s_XImats[504]
            //   T s_temp[1722]
            //   bytes s_linalg_smem[GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()]
            extern __shared__ __align__(16) unsigned char s_arena[];
            size_t s_arena_offset = 0;
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_q_qd = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(14);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_dc_du = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(98);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_vaf = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(126);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_XImats = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(504);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_temp = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(1722);
            int *s_topology_helpers = nullptr;
            unsigned char *s_linalg_smem = nullptr;
            if (static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) > 0) {
                s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
                s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
                s_arena_offset += static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());
            }
            #ifdef GRID_CUDA_DEBUG_LAYOUT
            assert(s_arena_offset == grid_shared_arena_bytes<T>(2464, 0, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()));
            #endif
            (void)s_arena_offset;
            T *d_temp_spill = nullptr; (void)d_temp_spill;
            T *s_q = s_q_qd; T *s_qd = &s_q_qd[7];
            for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
                // load to shared mem
                const T *d_q_qd_k = &d_q_qd[k*stride_q_qd];
                for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
                    s_q_qd[ind] = d_q_qd_k[ind];
                }
                __syncthreads();
                // compute — the orchestration inner owns its s_temp pool placement
                inverse_dynamics_gradient_device<T, true, false>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_topology_helpers, s_temp, nullptr, nullptr, d_robotModel, d_f_ext, gravity);
                __syncthreads();
                // save down to global
                T *d_dc_du_k = &d_dc_du[k*98];
                for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
                    d_dc_du_k[ind] = s_dc_du[ind];
                }
                __syncthreads();
            }
        }
        else if constexpr (RESOURCE_TIER == TIER_MINIMAL) {
            // GRID shared arena layout
            //   T s_q_qd[14]
            //   T s_dc_du[98]
            //   T s_vaf[126]
            //   T s_XImats[504]
            //   bytes s_linalg_smem[GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()]
            extern __shared__ __align__(16) unsigned char s_arena[];
            size_t s_arena_offset = 0;
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_q_qd = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(14);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_dc_du = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(98);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_vaf = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(126);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_XImats = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(504);
            T *s_temp = nullptr;
            int *s_topology_helpers = nullptr;
            unsigned char *s_linalg_smem = nullptr;
            if (static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) > 0) {
                s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
                s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
                s_arena_offset += static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());
            }
            #ifdef GRID_CUDA_DEBUG_LAYOUT
            assert(s_arena_offset == grid_shared_arena_bytes<T>(742, 0, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()));
            #endif
            (void)s_arena_offset;
            T *d_temp_spill = nullptr; (void)d_temp_spill;
            T *s_q = s_q_qd; T *s_qd = &s_q_qd[7];
            for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
                // load to shared mem
                const T *d_q_qd_k = &d_q_qd[k*stride_q_qd];
                for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
                    s_q_qd[ind] = d_q_qd_k[ind];
                }
                __syncthreads();
                // compute — the orchestration inner owns its s_temp pool placement
                inverse_dynamics_gradient_device<T, false, false>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_topology_helpers, s_temp, reinterpret_cast<T *>(&d_workspace[grid_workspace_slot()*GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()]), nullptr, d_robotModel, d_f_ext, gravity);
                __syncthreads();
                // save down to global
                T *d_dc_du_k = &d_dc_du[k*98];
                for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
                    d_dc_du_k[ind] = s_dc_du[ind];
                }
                __syncthreads();
            }
        }
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_QDD_FLAG = false, bool USE_COMPRESSED_MEM = false, gridDataKind KIND = GRID_DATA_ALL, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>
    __host__
    void inverse_dynamics_gradient(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                                   const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_DYNAMICS, "inverse_dynamics_gradient requires all-data or dynamics gridData");
        // start code with memory transfer
        int stride_q_qd;
        if (USE_COMPRESSED_MEM) {stride_q_qd = 2*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd,hd_data->h_q_qd,stride_q_qd*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        else {stride_q_qd = 3*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        if (USE_QDD_FLAG) {gpuErrchk(cudaMemcpyAsync(hd_data->d_qdd,hd_data->h_qdd,NUM_JOINTS*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[1]));}
        gpuErrchkKernel();
        // then call the kernel
        gpuErrchk(grid_check_dynamic_shared_memory_bytes("inverse_dynamics_gradient", INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()));
        const int _grid_ws_n = (hd_data->workspace_timestep_slots > 0 && hd_data->workspace_timestep_slots < num_timesteps) ? hd_data->workspace_timestep_slots : num_timesteps;
        if (GRID_INVERSE_DYNAMICS_GRADIENT_USES_WORKSPACE_ANY_TIER) {gpuErrchk(grid_begin_l2_persisting(0, hd_data->d_workspace, GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()*static_cast<size_t>(_grid_ws_n)));}
        dim3 _ws_grid = block_dimms;
        if ((int)(_ws_grid.x*_ws_grid.y*_ws_grid.z) > _grid_ws_n) { _ws_grid = dim3(_grid_ws_n,1,1); }
        if (USE_QDD_FLAG) {
            if (USE_COMPRESSED_MEM) {inverse_dynamics_gradient_kernel<T, RESOURCE_TIER><<<_ws_grid,thread_dimms,INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()>>>(hd_data->d_dc_du,hd_data->d_workspace,hd_data->d_q_qd,stride_q_qd,hd_data->d_qdd, hd_data->d_f_ext,d_robotModel,gravity,num_timesteps);}
            else                    {inverse_dynamics_gradient_kernel<T, RESOURCE_TIER><<<_ws_grid,thread_dimms,INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()>>>(hd_data->d_dc_du,hd_data->d_workspace,hd_data->d_q_qd_u,stride_q_qd,hd_data->d_qdd, hd_data->d_f_ext,d_robotModel,gravity,num_timesteps);}
        }
        else {
            if (USE_COMPRESSED_MEM) {inverse_dynamics_gradient_kernel<T, RESOURCE_TIER><<<_ws_grid,thread_dimms,INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()>>>(hd_data->d_dc_du,hd_data->d_workspace,hd_data->d_q_qd,stride_q_qd,hd_data->d_f_ext,d_robotModel,gravity,num_timesteps);}
            else                    {inverse_dynamics_gradient_kernel<T, RESOURCE_TIER><<<_ws_grid,thread_dimms,INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()>>>(hd_data->d_dc_du,hd_data->d_workspace,hd_data->d_q_qd_u,stride_q_qd,hd_data->d_f_ext,d_robotModel,gravity,num_timesteps);}
        }
        gpuErrchkKernel();
        if (GRID_INVERSE_DYNAMICS_GRADIENT_USES_WORKSPACE_ANY_TIER) {gpuErrchk(grid_end_l2_persisting(0));}
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_dc_du,hd_data->d_dc_du,2*NUM_VEL*NUM_VEL*num_timesteps*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchkKernel();
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_QDD_FLAG = false, bool USE_COMPRESSED_MEM = false, gridDataKind KIND = GRID_DATA_ALL, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>
    __host__
    void inverse_dynamics_gradient_single_timing(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                                                 const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_DYNAMICS, "inverse_dynamics_gradient requires all-data or dynamics gridData");
        // start code with memory transfer
        int stride_q_qd;
        if (USE_COMPRESSED_MEM) {stride_q_qd = 2*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd,hd_data->h_q_qd,stride_q_qd*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        else {stride_q_qd = 3*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        if (USE_QDD_FLAG) {gpuErrchk(cudaMemcpyAsync(hd_data->d_qdd,hd_data->h_qdd,NUM_JOINTS*sizeof(T),cudaMemcpyHostToDevice,streams[1]));}
        gpuErrchkKernel();
        // then call the kernel
        gpuErrchk(grid_check_dynamic_shared_memory_bytes("inverse_dynamics_gradient", INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()));
        if (GRID_INVERSE_DYNAMICS_GRADIENT_USES_WORKSPACE_ANY_TIER) {gpuErrchk(grid_begin_l2_persisting(0, hd_data->d_workspace, GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()));}
        struct timespec start, end; clock_gettime(CLOCK_MONOTONIC,&start);
        if (USE_QDD_FLAG) {
            if (USE_COMPRESSED_MEM) {inverse_dynamics_gradient_kernel_single_timing<T, RESOURCE_TIER><<<block_dimms,thread_dimms,INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()>>>(hd_data->d_dc_du,hd_data->d_workspace,hd_data->d_q_qd,stride_q_qd,hd_data->d_qdd, hd_data->d_f_ext,d_robotModel,gravity,num_timesteps);}
            else                    {inverse_dynamics_gradient_kernel_single_timing<T, RESOURCE_TIER><<<block_dimms,thread_dimms,INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()>>>(hd_data->d_dc_du,hd_data->d_workspace,hd_data->d_q_qd_u,stride_q_qd,hd_data->d_qdd, hd_data->d_f_ext,d_robotModel,gravity,num_timesteps);}
        }
        else {
            if (USE_COMPRESSED_MEM) {inverse_dynamics_gradient_kernel_single_timing<T, RESOURCE_TIER><<<block_dimms,thread_dimms,INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()>>>(hd_data->d_dc_du,hd_data->d_workspace,hd_data->d_q_qd,stride_q_qd,hd_data->d_f_ext,d_robotModel,gravity,num_timesteps);}
            else                    {inverse_dynamics_gradient_kernel_single_timing<T, RESOURCE_TIER><<<block_dimms,thread_dimms,INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()>>>(hd_data->d_dc_du,hd_data->d_workspace,hd_data->d_q_qd_u,stride_q_qd,hd_data->d_f_ext,d_robotModel,gravity,num_timesteps);}
        }
        gpuErrchkKernel();
        clock_gettime(CLOCK_MONOTONIC,&end);
        if (GRID_INVERSE_DYNAMICS_GRADIENT_USES_WORKSPACE_ANY_TIER) {gpuErrchk(grid_end_l2_persisting(0));}
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_dc_du,hd_data->d_dc_du,2*NUM_VEL*NUM_VEL*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchkKernel();
        printf("Single Call INVERSE_DYNAMICS_GRADIENT %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(num_timesteps));
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_QDD_FLAG = false, bool USE_COMPRESSED_MEM = false, gridDataKind KIND = GRID_DATA_ALL, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>
    __host__
    void inverse_dynamics_gradient_compute_only(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                                                const dim3 block_dimms, const dim3 thread_dimms) {
        static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_DYNAMICS, "inverse_dynamics_gradient requires all-data or dynamics gridData");
        int stride_q_qd = USE_COMPRESSED_MEM ? 2*NUM_JOINTS: 3*NUM_JOINTS;
        // then call the kernel
        gpuErrchk(grid_check_dynamic_shared_memory_bytes("inverse_dynamics_gradient", INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()));
        const int _grid_ws_n = (hd_data->workspace_timestep_slots > 0 && hd_data->workspace_timestep_slots < num_timesteps) ? hd_data->workspace_timestep_slots : num_timesteps;
        if (GRID_INVERSE_DYNAMICS_GRADIENT_USES_WORKSPACE_ANY_TIER) {gpuErrchk(grid_begin_l2_persisting(0, hd_data->d_workspace, GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()*static_cast<size_t>(_grid_ws_n)));}
        dim3 _ws_grid = block_dimms;
        if ((int)(_ws_grid.x*_ws_grid.y*_ws_grid.z) > _grid_ws_n) { _ws_grid = dim3(_grid_ws_n,1,1); }
        if (USE_QDD_FLAG) {
            if (USE_COMPRESSED_MEM) {inverse_dynamics_gradient_kernel<T, RESOURCE_TIER><<<_ws_grid,thread_dimms,INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()>>>(hd_data->d_dc_du,hd_data->d_workspace,hd_data->d_q_qd,stride_q_qd,hd_data->d_qdd, hd_data->d_f_ext,d_robotModel,gravity,num_timesteps);}
            else                    {inverse_dynamics_gradient_kernel<T, RESOURCE_TIER><<<_ws_grid,thread_dimms,INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()>>>(hd_data->d_dc_du,hd_data->d_workspace,hd_data->d_q_qd_u,stride_q_qd,hd_data->d_qdd, hd_data->d_f_ext,d_robotModel,gravity,num_timesteps);}
        }
        else {
            if (USE_COMPRESSED_MEM) {inverse_dynamics_gradient_kernel<T, RESOURCE_TIER><<<_ws_grid,thread_dimms,INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()>>>(hd_data->d_dc_du,hd_data->d_workspace,hd_data->d_q_qd,stride_q_qd,hd_data->d_f_ext,d_robotModel,gravity,num_timesteps);}
            else                    {inverse_dynamics_gradient_kernel<T, RESOURCE_TIER><<<_ws_grid,thread_dimms,INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()>>>(hd_data->d_dc_du,hd_data->d_workspace,hd_data->d_q_qd_u,stride_q_qd,hd_data->d_f_ext,d_robotModel,gravity,num_timesteps);}
        }
        gpuErrchkKernel();
        if (GRID_INVERSE_DYNAMICS_GRADIENT_USES_WORKSPACE_ANY_TIER) {gpuErrchk(grid_end_l2_persisting(0));}
    }

    /**
     * forward_dynamics_gradient orchestration as a single inner-owns-placement device function
     *
     * Notes:
     *   Uses the fd/du = -Minv*id/du trick (Carpentier & Mansard 'Analytical Derivatives of Rigid Body Dynamics Algorithms')
     *   Owns the s_temp pool placement; the repoint covers every consumer below (incl. the XImats helper's sincos scratch and minv's F-region)
     *
     * @param s_df_du is the output buffer (caller places); size 2*NUM_VEL*NUM_VEL = 98
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param s_u is the vector of input torques
     * @param s_vaf is the id intermediate band (caller places); size 18*NUM_JOINTS = 126
     * @param s_dc_du is the inverse_dynamics_gradient output band (caller places); size 2*NUM_VEL*NUM_VEL = 98
     * @param s_qdd is the joint-accel scratch (caller places); size NUM_JOINTS = 7
     * @param s_Minv is the mass-matrix scratch (caller places); size NUM_VEL*NUM_VEL = 49
     * @param s_temp is the shared scratch pool (used when SCRATCH_IN_SMEM)
     * @param d_workspace is the global scratch pool (used when !SCRATCH_IN_SMEM)
     * @param d_temp_spill is the inverse_dynamics_gradient da_df band spill region (used when USE_DA_DF_SPILL)
     * @param s_XImats is the (shared) memory holding the updated XI matricies for the given s_q
     * @param s_topology_helpers is the (shared) memory location for the topology_helpers (nullptr/unused for serial chains with identical Ss)
     * @param d_f_ext is the (optional) GLOBAL external forces, body-major 6*NUM_BODIES local-frame, or nullptr
     * @param d_robotModel holds XImats/topology; gravity is the gravity constant
     */
    template <typename T, bool SCRATCH_IN_SMEM = true, bool USE_DA_DF_SPILL = false>
    __device__ __forceinline__
    void forward_dynamics_gradient_device(T *s_df_du, const T *s_q, const T *s_qd, const T *s_u, T *s_vaf, T *s_dc_du, T *s_qdd, T *s_Minv, T *s_XImats, int *s_topology_helpers, T *s_temp, T *d_workspace, T *d_temp_spill, const robotModel<T> *d_robotModel, T *d_f_ext, const T gravity) {
        if constexpr(!SCRATCH_IN_SMEM){ s_temp = d_workspace; } else { (void)d_workspace; }
        load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
        minv_inner<T, true>(s_Minv, s_q, s_XImats, s_topology_helpers, s_temp, nullptr);
        inverse_dynamics_inner<T>(s_temp, s_vaf, s_q, s_qd, s_XImats, s_topology_helpers, &s_temp[7], d_f_ext, gravity);
        forward_dynamics_finish<T>(s_qdd, s_u, s_temp, s_Minv);
        __syncthreads();
        inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_qdd, s_XImats, s_topology_helpers, s_temp, d_f_ext, gravity);
        inverse_dynamics_gradient_inner<T, USE_DA_DF_SPILL>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_topology_helpers, s_temp, d_temp_spill, gravity);
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
            int row = ind % 7; int dc_col_offset = ind - row;
            // account for the fact that Minv is an SYMMETRIC_UPPER triangular matrix
            T val = static_cast<T>(0);
            for(int col = 0; col < 7; col++) {
                int index = (row <= col) * (col * 7 + row) + (row > col) * (row * 7 + col);
                val += s_Minv[index] * s_dc_du[dc_col_offset + col];
            }
            s_df_du[ind] = -val;
        }
    }

    /**
     * forward_dynamics_gradient orchestration as a single inner-owns-placement device function
     *
     * Notes:
     *   Uses the fd/du = -Minv*id/du trick (Carpentier & Mansard 'Analytical Derivatives of Rigid Body Dynamics Algorithms')
     *   Owns the s_temp pool placement; the repoint covers every consumer below (incl. the XImats helper's sincos scratch and minv's F-region)
     *
     * @param s_df_du is the output buffer (caller places); size 2*NUM_VEL*NUM_VEL = 98
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param s_qdd is the vector of joint accelerations (input)
     * @param s_Minv is the mass matrix (input)
     * @param s_vaf is the id intermediate band (caller places); size 18*NUM_JOINTS = 126
     * @param s_dc_du is the inverse_dynamics_gradient output band (caller places); size 2*NUM_VEL*NUM_VEL = 98
     * @param s_qdd is the joint-accel scratch (caller places); size NUM_JOINTS = 7
     * @param s_Minv is the mass-matrix scratch (caller places); size NUM_VEL*NUM_VEL = 49
     * @param s_temp is the shared scratch pool (used when SCRATCH_IN_SMEM)
     * @param d_workspace is the global scratch pool (used when !SCRATCH_IN_SMEM)
     * @param d_temp_spill is the inverse_dynamics_gradient da_df band spill region (used when USE_DA_DF_SPILL)
     * @param s_XImats is the (shared) memory holding the updated XI matricies for the given s_q
     * @param s_topology_helpers is the (shared) memory location for the topology_helpers (nullptr/unused for serial chains with identical Ss)
     * @param d_f_ext is the (optional) GLOBAL external forces, body-major 6*NUM_BODIES local-frame, or nullptr
     * @param d_robotModel holds XImats/topology; gravity is the gravity constant
     */
    template <typename T, bool SCRATCH_IN_SMEM = true, bool USE_DA_DF_SPILL = false>
    __device__ __forceinline__
    void forward_dynamics_gradient_device_qdd(T *s_df_du, const T *s_q, const T *s_qd, const T *s_qdd, const T *s_Minv, T *s_vaf, T *s_dc_du, const T *s_qdd_unused, const T *s_Minv_unused, T *s_XImats, int *s_topology_helpers, T *s_temp, T *d_workspace, T *d_temp_spill, const robotModel<T> *d_robotModel, T *d_f_ext, const T gravity) {
        (void)s_qdd_unused; (void)s_Minv_unused;
        if constexpr(!SCRATCH_IN_SMEM){ s_temp = d_workspace; } else { (void)d_workspace; }
        load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
        inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_qdd, s_XImats, s_topology_helpers, s_temp, d_f_ext, gravity);
        inverse_dynamics_gradient_inner<T, USE_DA_DF_SPILL>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_topology_helpers, s_temp, d_temp_spill, gravity);
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
            int row = ind % 7; int dc_col_offset = ind - row;
            // account for the fact that Minv is an SYMMETRIC_UPPER triangular matrix
            T val = static_cast<T>(0);
            for(int col = 0; col < 7; col++) {
                int index = (row <= col) * (col * 7 + row) + (row > col) * (row * 7 + col);
                val += s_Minv[index] * s_dc_du[dc_col_offset + col];
            }
            s_df_du[ind] = -val;
        }
    }

    /**
     * Computes the gradient of forward dynamics
     *
     * @param d_df_du is a pointer to memory for the final result of size 2*NUM_VEL*NUM_VEL = 98
     * @param d_q_dq is the vector of joint positions and velocities
     * @param stride_q_qd is the stide between each q, qd
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param d_f_ext is the (optional) GLOBAL external forces, body-major 6*NUM_BODIES local-frame, or nullptr
     * @param d_qdd is the vector of joint accelerations
     * @param d_Minv is the mass matrix
     * @param gravity is the gravity constant
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER, bool MUJOCO_OUTPUT = false>
    __global__
    __launch_bounds__(tier_max_threads<RESOURCE_TIER>())
    void forward_dynamics_gradient_kernel_single_timing(T *d_df_du, unsigned char *d_workspace, const T *d_q_qd, const int stride_q_qd, const T *d_qdd, const T *d_Minv, T *d_f_ext, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        if constexpr (RESOURCE_TIER == TIER_SHARED) {
            // GRID shared arena layout
            //   T s_q_qd[14]
            //   T s_dc_du[98]
            //   T s_vaf[126]
            //   T s_qdd[7]
            //   T s_Minv[49]
            //   T s_XImats[504]
            //   T s_temp[1722]
            //   bytes s_linalg_smem[GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()]
            extern __shared__ __align__(16) unsigned char s_arena[];
            size_t s_arena_offset = 0;
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_q_qd = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(14);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_dc_du = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(98);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_vaf = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(126);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_qdd = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(7);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_Minv = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(49);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_XImats = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(504);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_temp = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(1722);
            int *s_topology_helpers = nullptr;
            unsigned char *s_linalg_smem = nullptr;
            if (static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) > 0) {
                s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
                s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
                s_arena_offset += static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());
            }
            #ifdef GRID_CUDA_DEBUG_LAYOUT
            assert(s_arena_offset == grid_shared_arena_bytes<T>(2520, 0, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()));
            #endif
            (void)s_arena_offset;
            T *d_temp_spill = nullptr; (void)d_temp_spill;
            T *s_q = s_q_qd; T *s_qd = &s_q_qd[7];
            // load to shared mem
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
                s_q_qd[ind] = d_q_qd[ind];
            }
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                s_qdd[ind] = d_qdd[ind];
            }
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 49; ind += blockDim.x*blockDim.y){
                s_Minv[ind] = d_Minv[ind];
            }
            __syncthreads();
            // compute with NUM_TIMESTEPS as NUM_REPS for timing
            for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
                // anti-LICM: volatile reload of inputs each rep
                for(int _aopt_i = threadIdx.x + threadIdx.y*blockDim.x; _aopt_i < 14; _aopt_i += blockDim.x*blockDim.y){
                    reinterpret_cast<volatile T *>(s_q_qd)[_aopt_i] = reinterpret_cast<const volatile T *>(d_q_qd)[_aopt_i];
                }
                for(int _aopt_i = threadIdx.x + threadIdx.y*blockDim.x; _aopt_i < 7; _aopt_i += blockDim.x*blockDim.y){
                    reinterpret_cast<volatile T *>(s_qdd)[_aopt_i] = reinterpret_cast<const volatile T *>(d_qdd)[_aopt_i];
                }
                for(int _aopt_i = threadIdx.x + threadIdx.y*blockDim.x; _aopt_i < 49; _aopt_i += blockDim.x*blockDim.y){
                    reinterpret_cast<volatile T *>(s_Minv)[_aopt_i] = reinterpret_cast<const volatile T *>(d_Minv)[_aopt_i];
                }
                __syncthreads();
                // anti-LICM (1/2): stomp one input slot with `rep`
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
                    reinterpret_cast<volatile T *>(s_q_qd)[rep % (14)] = static_cast<T>(rep);
                    reinterpret_cast<volatile T *>(s_qdd)[rep % (7)] = static_cast<T>(rep);
                    reinterpret_cast<volatile T *>(s_Minv)[rep % (49)] = static_cast<T>(rep);
                }
                __syncthreads();
                forward_dynamics_gradient_device_qdd<T, true, false>(s_temp, s_q, s_qd, s_qdd, s_Minv, s_vaf, s_dc_du, s_qdd, s_Minv, s_XImats, s_topology_helpers, s_temp, nullptr, nullptr, d_robotModel, d_f_ext, gravity);
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) { reinterpret_cast<volatile T *>(d_df_du)[rep & 63] = s_temp[rep & 63]; }
            }
            // save down to global
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
                d_df_du[ind] = s_temp[ind];
            }
            __syncthreads();
        }
        else if constexpr (RESOURCE_TIER == TIER_LITE) {
            // GRID shared arena layout
            //   T s_q_qd[14]
            //   T s_dc_du[98]
            //   T s_vaf[126]
            //   T s_qdd[7]
            //   T s_Minv[49]
            //   T s_XImats[504]
            //   T s_temp[1722]
            //   bytes s_linalg_smem[GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()]
            extern __shared__ __align__(16) unsigned char s_arena[];
            size_t s_arena_offset = 0;
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_q_qd = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(14);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_dc_du = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(98);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_vaf = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(126);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_qdd = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(7);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_Minv = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(49);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_XImats = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(504);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_temp = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(1722);
            int *s_topology_helpers = nullptr;
            unsigned char *s_linalg_smem = nullptr;
            if (static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) > 0) {
                s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
                s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
                s_arena_offset += static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());
            }
            #ifdef GRID_CUDA_DEBUG_LAYOUT
            assert(s_arena_offset == grid_shared_arena_bytes<T>(2520, 0, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()));
            #endif
            (void)s_arena_offset;
            T *d_temp_spill = nullptr; (void)d_temp_spill;
            T *s_q = s_q_qd; T *s_qd = &s_q_qd[7];
            // load to shared mem
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
                s_q_qd[ind] = d_q_qd[ind];
            }
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                s_qdd[ind] = d_qdd[ind];
            }
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 49; ind += blockDim.x*blockDim.y){
                s_Minv[ind] = d_Minv[ind];
            }
            __syncthreads();
            // compute with NUM_TIMESTEPS as NUM_REPS for timing
            for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
                // anti-LICM: volatile reload of inputs each rep
                for(int _aopt_i = threadIdx.x + threadIdx.y*blockDim.x; _aopt_i < 14; _aopt_i += blockDim.x*blockDim.y){
                    reinterpret_cast<volatile T *>(s_q_qd)[_aopt_i] = reinterpret_cast<const volatile T *>(d_q_qd)[_aopt_i];
                }
                for(int _aopt_i = threadIdx.x + threadIdx.y*blockDim.x; _aopt_i < 7; _aopt_i += blockDim.x*blockDim.y){
                    reinterpret_cast<volatile T *>(s_qdd)[_aopt_i] = reinterpret_cast<const volatile T *>(d_qdd)[_aopt_i];
                }
                for(int _aopt_i = threadIdx.x + threadIdx.y*blockDim.x; _aopt_i < 49; _aopt_i += blockDim.x*blockDim.y){
                    reinterpret_cast<volatile T *>(s_Minv)[_aopt_i] = reinterpret_cast<const volatile T *>(d_Minv)[_aopt_i];
                }
                __syncthreads();
                // anti-LICM (1/2): stomp one input slot with `rep`
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
                    reinterpret_cast<volatile T *>(s_q_qd)[rep % (14)] = static_cast<T>(rep);
                    reinterpret_cast<volatile T *>(s_qdd)[rep % (7)] = static_cast<T>(rep);
                    reinterpret_cast<volatile T *>(s_Minv)[rep % (49)] = static_cast<T>(rep);
                }
                __syncthreads();
                forward_dynamics_gradient_device_qdd<T, true, false>(s_temp, s_q, s_qd, s_qdd, s_Minv, s_vaf, s_dc_du, s_qdd, s_Minv, s_XImats, s_topology_helpers, s_temp, nullptr, nullptr, d_robotModel, d_f_ext, gravity);
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) { reinterpret_cast<volatile T *>(d_df_du)[rep & 63] = s_temp[rep & 63]; }
            }
            // save down to global
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
                d_df_du[ind] = s_temp[ind];
            }
            __syncthreads();
        }
        else if constexpr (RESOURCE_TIER == TIER_MINIMAL) {
            // GRID shared arena layout
            //   T s_q_qd[14]
            //   T s_vaf[126]
            //   T s_qdd[7]
            //   T s_XImats[504]
            //   bytes s_linalg_smem[GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()]
            extern __shared__ __align__(16) unsigned char s_arena[];
            size_t s_arena_offset = 0;
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_q_qd = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(14);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_vaf = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(126);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_qdd = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(7);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_XImats = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(504);
            T *s_temp = nullptr;
            int *s_topology_helpers = nullptr;
            unsigned char *s_linalg_smem = nullptr;
            if (static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) > 0) {
                s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
                s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
                s_arena_offset += static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());
            }
            #ifdef GRID_CUDA_DEBUG_LAYOUT
            assert(s_arena_offset == grid_shared_arena_bytes<T>(651, 0, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()));
            #endif
            (void)s_arena_offset;
            T *s_dc_du; T *s_Minv;  // repointed to the L2-pinned SO band (output spill) per timing branch
            T *d_temp_spill = nullptr; (void)d_temp_spill;
            T *s_q = s_q_qd; T *s_qd = &s_q_qd[7];
            // load to shared mem
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
                s_q_qd[ind] = d_q_qd[ind];
            }
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                s_qdd[ind] = d_qdd[ind];
            }
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 49; ind += blockDim.x*blockDim.y){
                s_Minv[ind] = d_Minv[ind];
            }
            __syncthreads();
            T *d_df_du_k = d_df_du;
            s_dc_du = reinterpret_cast<T *>(&d_workspace[GRID_SO_WORKSPACE_TEMP_OFFSET_BYTES<T>()]); s_Minv = &s_dc_du[98];
            // compute with NUM_TIMESTEPS as NUM_REPS for timing
            for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
                // anti-LICM: volatile reload of inputs each rep
                for(int _aopt_i = threadIdx.x + threadIdx.y*blockDim.x; _aopt_i < 14; _aopt_i += blockDim.x*blockDim.y){
                    reinterpret_cast<volatile T *>(s_q_qd)[_aopt_i] = reinterpret_cast<const volatile T *>(d_q_qd)[_aopt_i];
                }
                for(int _aopt_i = threadIdx.x + threadIdx.y*blockDim.x; _aopt_i < 7; _aopt_i += blockDim.x*blockDim.y){
                    reinterpret_cast<volatile T *>(s_qdd)[_aopt_i] = reinterpret_cast<const volatile T *>(d_qdd)[_aopt_i];
                }
                for(int _aopt_i = threadIdx.x + threadIdx.y*blockDim.x; _aopt_i < 49; _aopt_i += blockDim.x*blockDim.y){
                    reinterpret_cast<volatile T *>(s_Minv)[_aopt_i] = reinterpret_cast<const volatile T *>(d_Minv)[_aopt_i];
                }
                __syncthreads();
                // anti-LICM (1/2): stomp one input slot with `rep`
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
                    reinterpret_cast<volatile T *>(s_q_qd)[rep % (14)] = static_cast<T>(rep);
                    reinterpret_cast<volatile T *>(s_qdd)[rep % (7)] = static_cast<T>(rep);
                    reinterpret_cast<volatile T *>(s_Minv)[rep % (49)] = static_cast<T>(rep);
                }
                __syncthreads();
                forward_dynamics_gradient_device_qdd<T, false, false>(d_df_du_k, s_q, s_qd, s_qdd, s_Minv, s_vaf, s_dc_du, s_qdd, s_Minv, s_XImats, s_topology_helpers, s_temp, reinterpret_cast<T *>(d_workspace), nullptr, d_robotModel, d_f_ext, gravity);
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) { reinterpret_cast<volatile T *>(d_df_du)[rep & 63] = d_df_du_k[rep & 63]; }
            }
        }
    }

    /**
     * Computes the gradient of forward dynamics
     *
     * @param d_df_du is a pointer to memory for the final result of size 2*NUM_VEL*NUM_VEL = 98
     * @param d_q_dq is the vector of joint positions and velocities
     * @param stride_q_qd is the stide between each q, qd
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param d_f_ext is the (optional) GLOBAL external forces, body-major 6*NUM_BODIES local-frame, or nullptr
     * @param d_qdd is the vector of joint accelerations
     * @param d_Minv is the mass matrix
     * @param gravity is the gravity constant
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER, bool MUJOCO_OUTPUT = false>
    __global__
    __launch_bounds__(tier_max_threads<RESOURCE_TIER>())
    void forward_dynamics_gradient_kernel(T *d_df_du, unsigned char *d_workspace, const T *d_q_qd, const int stride_q_qd, const T *d_qdd, const T *d_Minv, T *d_f_ext, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        if constexpr (RESOURCE_TIER == TIER_SHARED) {
            // GRID shared arena layout
            //   T s_q_qd[14]
            //   T s_dc_du[98]
            //   T s_vaf[126]
            //   T s_qdd[7]
            //   T s_Minv[49]
            //   T s_XImats[504]
            //   T s_temp[1722]
            //   bytes s_linalg_smem[GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()]
            extern __shared__ __align__(16) unsigned char s_arena[];
            size_t s_arena_offset = 0;
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_q_qd = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(14);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_dc_du = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(98);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_vaf = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(126);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_qdd = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(7);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_Minv = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(49);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_XImats = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(504);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_temp = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(1722);
            int *s_topology_helpers = nullptr;
            unsigned char *s_linalg_smem = nullptr;
            if (static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) > 0) {
                s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
                s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
                s_arena_offset += static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());
            }
            #ifdef GRID_CUDA_DEBUG_LAYOUT
            assert(s_arena_offset == grid_shared_arena_bytes<T>(2520, 0, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()));
            #endif
            (void)s_arena_offset;
            T *d_temp_spill = nullptr; (void)d_temp_spill;
            T *s_q = s_q_qd; T *s_qd = &s_q_qd[7];
            for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
                // load to shared mem
                const T *d_q_qd_k = &d_q_qd[k*stride_q_qd];
                for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
                    s_q_qd[ind] = d_q_qd_k[ind];
                }
                const T *d_qdd_k = &d_qdd[k*7];
                for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                    s_qdd[ind] = d_qdd_k[ind];
                }
                const T *d_Minv_k = &d_Minv[k*49];
                for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 49; ind += blockDim.x*blockDim.y){
                    s_Minv[ind] = d_Minv_k[ind];
                }
                __syncthreads();
                // compute — the orchestration inner owns its s_temp pool placement
                forward_dynamics_gradient_device_qdd<T, true, false>(s_temp, s_q, s_qd, s_qdd, s_Minv, s_vaf, s_dc_du, s_qdd, s_Minv, s_XImats, s_topology_helpers, s_temp, nullptr, nullptr, d_robotModel, d_f_ext, gravity);
                // save down to global
                T *d_df_du_k = &d_df_du[k*98];
                for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
                    d_df_du_k[ind] = s_temp[ind];
                }
                __syncthreads();
            }
        }
        else if constexpr (RESOURCE_TIER == TIER_LITE) {
            // GRID shared arena layout
            //   T s_q_qd[14]
            //   T s_dc_du[98]
            //   T s_vaf[126]
            //   T s_qdd[7]
            //   T s_Minv[49]
            //   T s_XImats[504]
            //   T s_temp[1722]
            //   bytes s_linalg_smem[GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()]
            extern __shared__ __align__(16) unsigned char s_arena[];
            size_t s_arena_offset = 0;
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_q_qd = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(14);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_dc_du = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(98);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_vaf = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(126);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_qdd = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(7);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_Minv = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(49);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_XImats = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(504);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_temp = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(1722);
            int *s_topology_helpers = nullptr;
            unsigned char *s_linalg_smem = nullptr;
            if (static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) > 0) {
                s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
                s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
                s_arena_offset += static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());
            }
            #ifdef GRID_CUDA_DEBUG_LAYOUT
            assert(s_arena_offset == grid_shared_arena_bytes<T>(2520, 0, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()));
            #endif
            (void)s_arena_offset;
            T *d_temp_spill = nullptr; (void)d_temp_spill;
            T *s_q = s_q_qd; T *s_qd = &s_q_qd[7];
            for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
                // load to shared mem
                const T *d_q_qd_k = &d_q_qd[k*stride_q_qd];
                for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
                    s_q_qd[ind] = d_q_qd_k[ind];
                }
                const T *d_qdd_k = &d_qdd[k*7];
                for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                    s_qdd[ind] = d_qdd_k[ind];
                }
                const T *d_Minv_k = &d_Minv[k*49];
                for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 49; ind += blockDim.x*blockDim.y){
                    s_Minv[ind] = d_Minv_k[ind];
                }
                __syncthreads();
                // compute — the orchestration inner owns its s_temp pool placement
                forward_dynamics_gradient_device_qdd<T, true, false>(s_temp, s_q, s_qd, s_qdd, s_Minv, s_vaf, s_dc_du, s_qdd, s_Minv, s_XImats, s_topology_helpers, s_temp, nullptr, nullptr, d_robotModel, d_f_ext, gravity);
                // save down to global
                T *d_df_du_k = &d_df_du[k*98];
                for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
                    d_df_du_k[ind] = s_temp[ind];
                }
                __syncthreads();
            }
        }
        else if constexpr (RESOURCE_TIER == TIER_MINIMAL) {
            // GRID shared arena layout
            //   T s_q_qd[14]
            //   T s_vaf[126]
            //   T s_qdd[7]
            //   T s_XImats[504]
            //   bytes s_linalg_smem[GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()]
            extern __shared__ __align__(16) unsigned char s_arena[];
            size_t s_arena_offset = 0;
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_q_qd = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(14);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_vaf = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(126);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_qdd = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(7);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_XImats = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(504);
            T *s_temp = nullptr;
            int *s_topology_helpers = nullptr;
            unsigned char *s_linalg_smem = nullptr;
            if (static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) > 0) {
                s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
                s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
                s_arena_offset += static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());
            }
            #ifdef GRID_CUDA_DEBUG_LAYOUT
            assert(s_arena_offset == grid_shared_arena_bytes<T>(651, 0, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()));
            #endif
            (void)s_arena_offset;
            T *s_dc_du; T *s_Minv;  // repointed to the L2-pinned SO band (output spill) per timing branch
            T *d_temp_spill = nullptr; (void)d_temp_spill;
            T *s_q = s_q_qd; T *s_qd = &s_q_qd[7];
            for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
                // load to shared mem
                const T *d_q_qd_k = &d_q_qd[k*stride_q_qd];
                for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
                    s_q_qd[ind] = d_q_qd_k[ind];
                }
                const T *d_qdd_k = &d_qdd[k*7];
                for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                    s_qdd[ind] = d_qdd_k[ind];
                }
                const T *d_Minv_k = &d_Minv[k*49];
                for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 49; ind += blockDim.x*blockDim.y){
                    s_Minv[ind] = d_Minv_k[ind];
                }
                __syncthreads();
                T *d_df_du_k = &d_df_du[k*98];
                s_dc_du = reinterpret_cast<T *>(&d_workspace[grid_workspace_slot()*GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>() + GRID_SO_WORKSPACE_TEMP_OFFSET_BYTES<T>()]); s_Minv = &s_dc_du[98];
                // compute — the orchestration inner owns its s_temp pool placement
                forward_dynamics_gradient_device_qdd<T, false, false>(d_df_du_k, s_q, s_qd, s_qdd, s_Minv, s_vaf, s_dc_du, s_qdd, s_Minv, s_XImats, s_topology_helpers, s_temp, reinterpret_cast<T *>(&d_workspace[grid_workspace_slot()*GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()]), nullptr, d_robotModel, d_f_ext, gravity);
            }
        }
    }

    /**
     * Computes the gradient of forward dynamics
     *
     * @param d_df_du is a pointer to memory for the final result of size 2*NUM_VEL*NUM_VEL = 98
     * @param d_q_dq is the vector of joint positions, velocities, and input torques
     * @param stride_q_qd_u is the stide between each q, qd, u
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param d_f_ext is the (optional) GLOBAL external forces, body-major 6*NUM_BODIES local-frame, or nullptr
     * @param gravity is the gravity constant
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER, bool MUJOCO_OUTPUT = false>
    __global__
    __launch_bounds__(tier_max_threads<RESOURCE_TIER>())
    void forward_dynamics_gradient_kernel_single_timing(T *d_df_du, unsigned char *d_workspace, const T *d_q_qd_u, const int stride_q_qd_u, T *d_f_ext, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        if constexpr (RESOURCE_TIER == TIER_SHARED) {
            // GRID shared arena layout
            //   T s_q_qd_u[21]
            //   T s_dc_du[98]
            //   T s_vaf[126]
            //   T s_qdd[7]
            //   T s_Minv[49]
            //   T s_XImats[504]
            //   T s_temp[1722]
            //   bytes s_linalg_smem[GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()]
            extern __shared__ __align__(16) unsigned char s_arena[];
            size_t s_arena_offset = 0;
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_q_qd_u = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(21);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_dc_du = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(98);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_vaf = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(126);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_qdd = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(7);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_Minv = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(49);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_XImats = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(504);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_temp = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(1722);
            int *s_topology_helpers = nullptr;
            unsigned char *s_linalg_smem = nullptr;
            if (static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) > 0) {
                s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
                s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
                s_arena_offset += static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());
            }
            #ifdef GRID_CUDA_DEBUG_LAYOUT
            assert(s_arena_offset == grid_shared_arena_bytes<T>(2527, 0, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()));
            #endif
            (void)s_arena_offset;
            T *d_temp_spill = nullptr; (void)d_temp_spill;
            T *s_q = s_q_qd_u; T *s_qd = &s_q_qd_u[7]; T *s_u = &s_q_qd_u[14];
            // load to shared mem
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 21; ind += blockDim.x*blockDim.y){
                s_q_qd_u[ind] = d_q_qd_u[ind];
            }
            __syncthreads();
            // compute with NUM_TIMESTEPS as NUM_REPS for timing
            for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
                // anti-LICM: volatile reload of inputs each rep
                for(int _aopt_i = threadIdx.x + threadIdx.y*blockDim.x; _aopt_i < 21; _aopt_i += blockDim.x*blockDim.y){
                    reinterpret_cast<volatile T *>(s_q_qd_u)[_aopt_i] = reinterpret_cast<const volatile T *>(d_q_qd_u)[_aopt_i];
                }
                __syncthreads();
                // anti-LICM (1/2): stomp one input slot with `rep`
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
                    reinterpret_cast<volatile T *>(s_q_qd_u)[rep % (21)] = static_cast<T>(rep);
                }
                __syncthreads();
                forward_dynamics_gradient_device<T, true, false>(s_temp, s_q, s_qd, s_u, s_vaf, s_dc_du, s_qdd, s_Minv, s_XImats, s_topology_helpers, s_temp, nullptr, nullptr, d_robotModel, d_f_ext, gravity);
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) { reinterpret_cast<volatile T *>(d_df_du)[rep & 63] = s_temp[rep & 63]; }
            }
            // save down to global
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
                d_df_du[ind] = s_temp[ind];
            }
            __syncthreads();
        }
        else if constexpr (RESOURCE_TIER == TIER_LITE) {
            // GRID shared arena layout
            //   T s_q_qd_u[21]
            //   T s_dc_du[98]
            //   T s_vaf[126]
            //   T s_qdd[7]
            //   T s_Minv[49]
            //   T s_XImats[504]
            //   T s_temp[1722]
            //   bytes s_linalg_smem[GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()]
            extern __shared__ __align__(16) unsigned char s_arena[];
            size_t s_arena_offset = 0;
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_q_qd_u = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(21);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_dc_du = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(98);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_vaf = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(126);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_qdd = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(7);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_Minv = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(49);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_XImats = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(504);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_temp = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(1722);
            int *s_topology_helpers = nullptr;
            unsigned char *s_linalg_smem = nullptr;
            if (static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) > 0) {
                s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
                s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
                s_arena_offset += static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());
            }
            #ifdef GRID_CUDA_DEBUG_LAYOUT
            assert(s_arena_offset == grid_shared_arena_bytes<T>(2527, 0, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()));
            #endif
            (void)s_arena_offset;
            T *d_temp_spill = nullptr; (void)d_temp_spill;
            T *s_q = s_q_qd_u; T *s_qd = &s_q_qd_u[7]; T *s_u = &s_q_qd_u[14];
            // load to shared mem
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 21; ind += blockDim.x*blockDim.y){
                s_q_qd_u[ind] = d_q_qd_u[ind];
            }
            __syncthreads();
            // compute with NUM_TIMESTEPS as NUM_REPS for timing
            for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
                // anti-LICM: volatile reload of inputs each rep
                for(int _aopt_i = threadIdx.x + threadIdx.y*blockDim.x; _aopt_i < 21; _aopt_i += blockDim.x*blockDim.y){
                    reinterpret_cast<volatile T *>(s_q_qd_u)[_aopt_i] = reinterpret_cast<const volatile T *>(d_q_qd_u)[_aopt_i];
                }
                __syncthreads();
                // anti-LICM (1/2): stomp one input slot with `rep`
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
                    reinterpret_cast<volatile T *>(s_q_qd_u)[rep % (21)] = static_cast<T>(rep);
                }
                __syncthreads();
                forward_dynamics_gradient_device<T, true, false>(s_temp, s_q, s_qd, s_u, s_vaf, s_dc_du, s_qdd, s_Minv, s_XImats, s_topology_helpers, s_temp, nullptr, nullptr, d_robotModel, d_f_ext, gravity);
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) { reinterpret_cast<volatile T *>(d_df_du)[rep & 63] = s_temp[rep & 63]; }
            }
            // save down to global
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
                d_df_du[ind] = s_temp[ind];
            }
            __syncthreads();
        }
        else if constexpr (RESOURCE_TIER == TIER_MINIMAL) {
            // GRID shared arena layout
            //   T s_q_qd_u[21]
            //   T s_vaf[126]
            //   T s_qdd[7]
            //   T s_XImats[504]
            //   bytes s_linalg_smem[GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()]
            extern __shared__ __align__(16) unsigned char s_arena[];
            size_t s_arena_offset = 0;
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_q_qd_u = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(21);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_vaf = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(126);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_qdd = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(7);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_XImats = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(504);
            T *s_temp = nullptr;
            int *s_topology_helpers = nullptr;
            unsigned char *s_linalg_smem = nullptr;
            if (static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) > 0) {
                s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
                s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
                s_arena_offset += static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());
            }
            #ifdef GRID_CUDA_DEBUG_LAYOUT
            assert(s_arena_offset == grid_shared_arena_bytes<T>(658, 0, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()));
            #endif
            (void)s_arena_offset;
            T *s_dc_du; T *s_Minv;  // repointed to the L2-pinned SO band (output spill) per timing branch
            T *d_temp_spill = nullptr; (void)d_temp_spill;
            T *s_q = s_q_qd_u; T *s_qd = &s_q_qd_u[7]; T *s_u = &s_q_qd_u[14];
            // load to shared mem
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 21; ind += blockDim.x*blockDim.y){
                s_q_qd_u[ind] = d_q_qd_u[ind];
            }
            __syncthreads();
            T *d_df_du_k = d_df_du;
            s_dc_du = reinterpret_cast<T *>(&d_workspace[GRID_SO_WORKSPACE_TEMP_OFFSET_BYTES<T>()]); s_Minv = &s_dc_du[98];
            // compute with NUM_TIMESTEPS as NUM_REPS for timing
            for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
                // anti-LICM: volatile reload of inputs each rep
                for(int _aopt_i = threadIdx.x + threadIdx.y*blockDim.x; _aopt_i < 21; _aopt_i += blockDim.x*blockDim.y){
                    reinterpret_cast<volatile T *>(s_q_qd_u)[_aopt_i] = reinterpret_cast<const volatile T *>(d_q_qd_u)[_aopt_i];
                }
                __syncthreads();
                // anti-LICM (1/2): stomp one input slot with `rep`
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
                    reinterpret_cast<volatile T *>(s_q_qd_u)[rep % (21)] = static_cast<T>(rep);
                }
                __syncthreads();
                forward_dynamics_gradient_device<T, false, false>(d_df_du_k, s_q, s_qd, s_u, s_vaf, s_dc_du, s_qdd, s_Minv, s_XImats, s_topology_helpers, s_temp, reinterpret_cast<T *>(d_workspace), nullptr, d_robotModel, d_f_ext, gravity);
                if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) { reinterpret_cast<volatile T *>(d_df_du)[rep & 63] = d_df_du_k[rep & 63]; }
            }
        }
    }

    /**
     * Computes the gradient of forward dynamics
     *
     * @param d_df_du is a pointer to memory for the final result of size 2*NUM_VEL*NUM_VEL = 98
     * @param d_q_dq is the vector of joint positions, velocities, and input torques
     * @param stride_q_qd_u is the stide between each q, qd, u
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param d_f_ext is the (optional) GLOBAL external forces, body-major 6*NUM_BODIES local-frame, or nullptr
     * @param gravity is the gravity constant
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER, bool MUJOCO_OUTPUT = false>
    __global__
    __launch_bounds__(tier_max_threads<RESOURCE_TIER>())
    void forward_dynamics_gradient_kernel(T *d_df_du, unsigned char *d_workspace, const T *d_q_qd_u, const int stride_q_qd_u, T *d_f_ext, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        if constexpr (RESOURCE_TIER == TIER_SHARED) {
            // GRID shared arena layout
            //   T s_q_qd_u[21]
            //   T s_dc_du[98]
            //   T s_vaf[126]
            //   T s_qdd[7]
            //   T s_Minv[49]
            //   T s_XImats[504]
            //   T s_temp[1722]
            //   bytes s_linalg_smem[GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()]
            extern __shared__ __align__(16) unsigned char s_arena[];
            size_t s_arena_offset = 0;
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_q_qd_u = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(21);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_dc_du = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(98);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_vaf = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(126);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_qdd = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(7);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_Minv = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(49);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_XImats = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(504);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_temp = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(1722);
            int *s_topology_helpers = nullptr;
            unsigned char *s_linalg_smem = nullptr;
            if (static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) > 0) {
                s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
                s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
                s_arena_offset += static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());
            }
            #ifdef GRID_CUDA_DEBUG_LAYOUT
            assert(s_arena_offset == grid_shared_arena_bytes<T>(2527, 0, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()));
            #endif
            (void)s_arena_offset;
            T *d_temp_spill = nullptr; (void)d_temp_spill;
            T *s_q = s_q_qd_u; T *s_qd = &s_q_qd_u[7]; T *s_u = &s_q_qd_u[14];
            for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
                // load to shared mem
                const T *d_q_qd_u_k = &d_q_qd_u[k*stride_q_qd_u];
                for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 21; ind += blockDim.x*blockDim.y){
                    s_q_qd_u[ind] = d_q_qd_u_k[ind];
                }
                __syncthreads();
                // compute — the orchestration inner owns its s_temp pool placement
                forward_dynamics_gradient_device<T, true, false>(s_temp, s_q, s_qd, s_u, s_vaf, s_dc_du, s_qdd, s_Minv, s_XImats, s_topology_helpers, s_temp, nullptr, nullptr, d_robotModel, d_f_ext, gravity);
                // save down to global
                T *d_df_du_k = &d_df_du[k*98];
                for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
                    d_df_du_k[ind] = s_temp[ind];
                }
                __syncthreads();
            }
        }
        else if constexpr (RESOURCE_TIER == TIER_LITE) {
            // GRID shared arena layout
            //   T s_q_qd_u[21]
            //   T s_dc_du[98]
            //   T s_vaf[126]
            //   T s_qdd[7]
            //   T s_Minv[49]
            //   T s_XImats[504]
            //   T s_temp[1722]
            //   bytes s_linalg_smem[GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()]
            extern __shared__ __align__(16) unsigned char s_arena[];
            size_t s_arena_offset = 0;
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_q_qd_u = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(21);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_dc_du = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(98);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_vaf = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(126);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_qdd = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(7);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_Minv = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(49);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_XImats = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(504);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_temp = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(1722);
            int *s_topology_helpers = nullptr;
            unsigned char *s_linalg_smem = nullptr;
            if (static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) > 0) {
                s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
                s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
                s_arena_offset += static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());
            }
            #ifdef GRID_CUDA_DEBUG_LAYOUT
            assert(s_arena_offset == grid_shared_arena_bytes<T>(2527, 0, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()));
            #endif
            (void)s_arena_offset;
            T *d_temp_spill = nullptr; (void)d_temp_spill;
            T *s_q = s_q_qd_u; T *s_qd = &s_q_qd_u[7]; T *s_u = &s_q_qd_u[14];
            for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
                // load to shared mem
                const T *d_q_qd_u_k = &d_q_qd_u[k*stride_q_qd_u];
                for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 21; ind += blockDim.x*blockDim.y){
                    s_q_qd_u[ind] = d_q_qd_u_k[ind];
                }
                __syncthreads();
                // compute — the orchestration inner owns its s_temp pool placement
                forward_dynamics_gradient_device<T, true, false>(s_temp, s_q, s_qd, s_u, s_vaf, s_dc_du, s_qdd, s_Minv, s_XImats, s_topology_helpers, s_temp, nullptr, nullptr, d_robotModel, d_f_ext, gravity);
                // save down to global
                T *d_df_du_k = &d_df_du[k*98];
                for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
                    d_df_du_k[ind] = s_temp[ind];
                }
                __syncthreads();
            }
        }
        else if constexpr (RESOURCE_TIER == TIER_MINIMAL) {
            // GRID shared arena layout
            //   T s_q_qd_u[21]
            //   T s_vaf[126]
            //   T s_qdd[7]
            //   T s_XImats[504]
            //   bytes s_linalg_smem[GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()]
            extern __shared__ __align__(16) unsigned char s_arena[];
            size_t s_arena_offset = 0;
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_q_qd_u = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(21);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_vaf = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(126);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_qdd = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(7);
            s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
            T *s_XImats = grid_arena_ptr<T>(s_arena, s_arena_offset);
            s_arena_offset += sizeof(T) * static_cast<size_t>(504);
            T *s_temp = nullptr;
            int *s_topology_helpers = nullptr;
            unsigned char *s_linalg_smem = nullptr;
            if (static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()) > 0) {
                s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
                s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
                s_arena_offset += static_cast<size_t>(GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());
            }
            #ifdef GRID_CUDA_DEBUG_LAYOUT
            assert(s_arena_offset == grid_shared_arena_bytes<T>(658, 0, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()));
            #endif
            (void)s_arena_offset;
            T *s_dc_du; T *s_Minv;  // repointed to the L2-pinned SO band (output spill) per timing branch
            T *d_temp_spill = nullptr; (void)d_temp_spill;
            T *s_q = s_q_qd_u; T *s_qd = &s_q_qd_u[7]; T *s_u = &s_q_qd_u[14];
            for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
                // load to shared mem
                const T *d_q_qd_u_k = &d_q_qd_u[k*stride_q_qd_u];
                for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 21; ind += blockDim.x*blockDim.y){
                    s_q_qd_u[ind] = d_q_qd_u_k[ind];
                }
                __syncthreads();
                T *d_df_du_k = &d_df_du[k*98];
                s_dc_du = reinterpret_cast<T *>(&d_workspace[grid_workspace_slot()*GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>() + GRID_SO_WORKSPACE_TEMP_OFFSET_BYTES<T>()]); s_Minv = &s_dc_du[98];
                // compute — the orchestration inner owns its s_temp pool placement
                forward_dynamics_gradient_device<T, false, false>(d_df_du_k, s_q, s_qd, s_u, s_vaf, s_dc_du, s_qdd, s_Minv, s_XImats, s_topology_helpers, s_temp, reinterpret_cast<T *>(&d_workspace[grid_workspace_slot()*GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()]), nullptr, d_robotModel, d_f_ext, gravity);
            }
        }
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_QDD_MINV_FLAG = false, gridDataKind KIND = GRID_DATA_ALL, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>
    __host__
    void forward_dynamics_gradient(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                          const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_DYNAMICS, "forward_dynamics_gradient requires all-data or dynamics gridData");
        int stride_q_qd= 3*NUM_JOINTS;
        // start code with memory transfer
        gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[0]));
        if (USE_QDD_MINV_FLAG) {
            gpuErrchk(cudaMemcpyAsync(hd_data->d_qdd,hd_data->h_qdd,NUM_JOINTS*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[1]));
            gpuErrchk(cudaMemcpyAsync(hd_data->d_Minv,hd_data->h_Minv,NUM_VEL*NUM_VEL*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[2]));
        }
        gpuErrchkKernel();
        // then call the kernel
        gpuErrchk(grid_check_dynamic_shared_memory_bytes("forward_dynamics_gradient", FORWARD_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()));
        const int _grid_ws_n = (hd_data->workspace_timestep_slots > 0 && hd_data->workspace_timestep_slots < num_timesteps) ? hd_data->workspace_timestep_slots : num_timesteps;
        if (GRID_FORWARD_DYNAMICS_GRADIENT_USES_WORKSPACE_ANY_TIER) {gpuErrchk(grid_begin_l2_persisting(0, hd_data->d_workspace, GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()*static_cast<size_t>(_grid_ws_n)));}
        dim3 _ws_grid = block_dimms;
        if ((int)(_ws_grid.x*_ws_grid.y*_ws_grid.z) > _grid_ws_n) { _ws_grid = dim3(_grid_ws_n,1,1); }
        if (USE_QDD_MINV_FLAG) {forward_dynamics_gradient_kernel<T, RESOURCE_TIER><<<_ws_grid,thread_dimms,FORWARD_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()>>>(hd_data->d_df_du,hd_data->d_workspace,hd_data->d_q_qd_u,stride_q_qd,hd_data->d_qdd, hd_data->d_Minv, hd_data->d_f_ext,d_robotModel,gravity,num_timesteps);}
        else {forward_dynamics_gradient_kernel<T, RESOURCE_TIER><<<_ws_grid,thread_dimms,FORWARD_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()>>>(hd_data->d_df_du,hd_data->d_workspace,hd_data->d_q_qd_u,stride_q_qd,hd_data->d_f_ext,d_robotModel,gravity,num_timesteps);}
        gpuErrchkKernel();
        if (GRID_FORWARD_DYNAMICS_GRADIENT_USES_WORKSPACE_ANY_TIER) {gpuErrchk(grid_end_l2_persisting(0));}
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_df_du,hd_data->d_df_du,2*NUM_VEL*NUM_VEL*num_timesteps*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchkKernel();
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_QDD_MINV_FLAG = false, gridDataKind KIND = GRID_DATA_ALL, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>
    __host__
    void forward_dynamics_gradient_single_timing(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                                        const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_DYNAMICS, "forward_dynamics_gradient requires all-data or dynamics gridData");
        int stride_q_qd= 3*NUM_JOINTS;
        // start code with memory transfer
        gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd*sizeof(T),cudaMemcpyHostToDevice,streams[0]));
        if (USE_QDD_MINV_FLAG) {
            gpuErrchk(cudaMemcpyAsync(hd_data->d_qdd,hd_data->h_qdd,NUM_JOINTS*sizeof(T),cudaMemcpyHostToDevice,streams[1]));
            gpuErrchk(cudaMemcpyAsync(hd_data->d_Minv,hd_data->h_Minv,NUM_VEL*NUM_VEL*sizeof(T),cudaMemcpyHostToDevice,streams[2]));
        }
        gpuErrchkKernel();
        // then call the kernel
        gpuErrchk(grid_check_dynamic_shared_memory_bytes("forward_dynamics_gradient", FORWARD_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()));
        if (GRID_FORWARD_DYNAMICS_GRADIENT_USES_WORKSPACE_ANY_TIER) {gpuErrchk(grid_begin_l2_persisting(0, hd_data->d_workspace, GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()));}
        struct timespec start, end; clock_gettime(CLOCK_MONOTONIC,&start);
        if (USE_QDD_MINV_FLAG) {forward_dynamics_gradient_kernel_single_timing<T, RESOURCE_TIER><<<block_dimms,thread_dimms,FORWARD_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()>>>(hd_data->d_df_du,hd_data->d_workspace,hd_data->d_q_qd_u,stride_q_qd,hd_data->d_qdd, hd_data->d_Minv, hd_data->d_f_ext,d_robotModel,gravity,num_timesteps);}
        else {forward_dynamics_gradient_kernel_single_timing<T, RESOURCE_TIER><<<block_dimms,thread_dimms,FORWARD_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()>>>(hd_data->d_df_du,hd_data->d_workspace,hd_data->d_q_qd_u,stride_q_qd,hd_data->d_f_ext,d_robotModel,gravity,num_timesteps);}
        gpuErrchkKernel();
        clock_gettime(CLOCK_MONOTONIC,&end);
        if (GRID_FORWARD_DYNAMICS_GRADIENT_USES_WORKSPACE_ANY_TIER) {gpuErrchk(grid_end_l2_persisting(0));}
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_df_du,hd_data->d_df_du,2*NUM_VEL*NUM_VEL*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchkKernel();
        printf("Single Call FORWARD_DYNAMICS_GRADIENT %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(num_timesteps));
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_QDD_MINV_FLAG = false, gridDataKind KIND = GRID_DATA_ALL, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>
    __host__
    void forward_dynamics_gradient_compute_only(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                                       const dim3 block_dimms, const dim3 thread_dimms) {
        static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_DYNAMICS, "forward_dynamics_gradient requires all-data or dynamics gridData");
        int stride_q_qd= 3*NUM_JOINTS;
        // then call the kernel
        gpuErrchk(grid_check_dynamic_shared_memory_bytes("forward_dynamics_gradient", FORWARD_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()));
        const int _grid_ws_n = (hd_data->workspace_timestep_slots > 0 && hd_data->workspace_timestep_slots < num_timesteps) ? hd_data->workspace_timestep_slots : num_timesteps;
        if (GRID_FORWARD_DYNAMICS_GRADIENT_USES_WORKSPACE_ANY_TIER) {gpuErrchk(grid_begin_l2_persisting(0, hd_data->d_workspace, GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()*static_cast<size_t>(_grid_ws_n)));}
        dim3 _ws_grid = block_dimms;
        if ((int)(_ws_grid.x*_ws_grid.y*_ws_grid.z) > _grid_ws_n) { _ws_grid = dim3(_grid_ws_n,1,1); }
        if (USE_QDD_MINV_FLAG) {forward_dynamics_gradient_kernel<T, RESOURCE_TIER><<<_ws_grid,thread_dimms,FORWARD_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()>>>(hd_data->d_df_du,hd_data->d_workspace,hd_data->d_q_qd_u,stride_q_qd,hd_data->d_qdd, hd_data->d_Minv, hd_data->d_f_ext,d_robotModel,gravity,num_timesteps);}
        else {forward_dynamics_gradient_kernel<T, RESOURCE_TIER><<<_ws_grid,thread_dimms,FORWARD_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()>>>(hd_data->d_df_du,hd_data->d_workspace,hd_data->d_q_qd_u,stride_q_qd,hd_data->d_f_ext,d_robotModel,gravity,num_timesteps);}
        gpuErrchkKernel();
        if (GRID_FORWARD_DYNAMICS_GRADIENT_USES_WORKSPACE_ANY_TIER) {gpuErrchk(grid_end_l2_persisting(0));}
    }

    // [centroidal] generalized_gravity/nonlinear_effects skipped: not requested (request 'generalized_gravity'/'nonlinear_effects').
    // [centroidal] com/ccrba/energy/dccrba/cmm_time_variation skipped: not requested.
    // [coriolis] coriolis_matrix skipped: not requested (request 'coriolis_matrix').
    /**
     * Run inverse dynamics, Minv, and forward dynamics in sequence
     *
     */
    template <typename T, gridDataKind KIND = GRID_DATA_ALL>
    __host__
    void dynamics_core(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                       const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_DYNAMICS, "dynamics_core requires all-data or dynamics gridData");
        inverse_dynamics<T,false,false,KIND>(hd_data,d_robotModel,gravity,num_timesteps,block_dimms,thread_dimms,streams);
        minv<T,false,KIND>(hd_data,d_robotModel,num_timesteps,block_dimms,thread_dimms,streams);
        forward_dynamics<T,KIND>(hd_data,d_robotModel,gravity,num_timesteps,block_dimms,thread_dimms,streams);
    }

    /**
     * Run inverse dynamics, Minv, and forward dynamics in sequence
     *
     */
    template <typename T, gridDataKind KIND = GRID_DATA_ALL>
    __host__
    void id_minv_fd(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                       const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_DYNAMICS, "id_minv_fd requires all-data or dynamics gridData");
        inverse_dynamics<T,false,false,KIND>(hd_data,d_robotModel,gravity,num_timesteps,block_dimms,thread_dimms,streams);
        minv<T,false,KIND>(hd_data,d_robotModel,num_timesteps,block_dimms,thread_dimms,streams);
        forward_dynamics<T,KIND>(hd_data,d_robotModel,gravity,num_timesteps,block_dimms,thread_dimms,streams);
    }

    /**
     * Run inverse dynamics and its first derivative in sequence
     *
     */
    template <typename T, gridDataKind KIND = GRID_DATA_ALL>
    __host__
    void id_and_id_gradient(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                       const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_DYNAMICS, "id_and_id_gradient requires all-data or dynamics gridData");
        inverse_dynamics<T,false,false,KIND>(hd_data,d_robotModel,gravity,num_timesteps,block_dimms,thread_dimms,streams);
        inverse_dynamics_gradient<T,false,false,KIND>(hd_data,d_robotModel,gravity,num_timesteps,block_dimms,thread_dimms,streams);
    }

    /**
     * Run forward dynamics and its first derivative in sequence
     *
     */
    template <typename T, gridDataKind KIND = GRID_DATA_ALL>
    __host__
    void fd_and_fd_gradient(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                       const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_DYNAMICS, "fd_and_fd_gradient requires all-data or dynamics gridData");
        forward_dynamics<T,KIND>(hd_data,d_robotModel,gravity,num_timesteps,block_dimms,thread_dimms,streams);
        forward_dynamics_gradient<T,false,KIND>(hd_data,d_robotModel,gravity,num_timesteps,block_dimms,thread_dimms,streams);
    }

    /**
     * Run inverse and forward dynamics gradients in sequence
     *
     */
    template <typename T, gridDataKind KIND = GRID_DATA_ALL>
    __host__
    void dynamics_gradients(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                       const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_DYNAMICS, "dynamics_gradients requires all-data or dynamics gridData");
        inverse_dynamics_gradient<T,false,false,KIND>(hd_data,d_robotModel,gravity,num_timesteps,block_dimms,thread_dimms,streams);
        forward_dynamics_gradient<T,false,KIND>(hd_data,d_robotModel,gravity,num_timesteps,block_dimms,thread_dimms,streams);
    }

    /**
     * Run all generated non-second-order dynamics wrappers in sequence
     *
     */
    template <typename T, gridDataKind KIND = GRID_DATA_ALL>
    __host__
    void all_dynamics(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                       const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_DYNAMICS, "all_dynamics requires all-data or dynamics gridData");
        inverse_dynamics<T,false,false,KIND>(hd_data,d_robotModel,gravity,num_timesteps,block_dimms,thread_dimms,streams);
        minv<T,false,KIND>(hd_data,d_robotModel,num_timesteps,block_dimms,thread_dimms,streams);
        forward_dynamics<T,KIND>(hd_data,d_robotModel,gravity,num_timesteps,block_dimms,thread_dimms,streams);
        inverse_dynamics_gradient<T,false,false,KIND>(hd_data,d_robotModel,gravity,num_timesteps,block_dimms,thread_dimms,streams);
        forward_dynamics_gradient<T,false,KIND>(hd_data,d_robotModel,gravity,num_timesteps,block_dimms,thread_dimms,streams);
    }

    /**
     * Run all generated non-second-order dynamics wrappers in sequence
     *
     */
    template <typename T, gridDataKind KIND = GRID_DATA_ALL>
    __host__
    void dynamics_only(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                       const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_DYNAMICS, "dynamics_only requires all-data or dynamics gridData");
        inverse_dynamics<T,false,false,KIND>(hd_data,d_robotModel,gravity,num_timesteps,block_dimms,thread_dimms,streams);
        minv<T,false,KIND>(hd_data,d_robotModel,num_timesteps,block_dimms,thread_dimms,streams);
        forward_dynamics<T,KIND>(hd_data,d_robotModel,gravity,num_timesteps,block_dimms,thread_dimms,streams);
        inverse_dynamics_gradient<T,false,false,KIND>(hd_data,d_robotModel,gravity,num_timesteps,block_dimms,thread_dimms,streams);
        forward_dynamics_gradient<T,false,KIND>(hd_data,d_robotModel,gravity,num_timesteps,block_dimms,thread_dimms,streams);
    }

    /**
     * Run all generated kinematics wrappers in sequence
     *
     */
    template <typename T, gridDataKind KIND = GRID_DATA_ALL>
    __host__
    void kinematics_only(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps,
                         const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_KINEMATICS, "kinematics_only requires all-data or kinematics gridData");
        end_effector_pose_EE<T,false,KIND>(hd_data,d_robotModel,num_timesteps,block_dimms,thread_dimms,streams);
        end_effector_pose_gradient_EE<T,false,KIND>(hd_data,d_robotModel,num_timesteps,block_dimms,thread_dimms,streams);
    }

    /**
     * Set MaxDynamicSharedMemorySize for every algorithm kernel (callable from any TU; idempotent). __forceinline__ is REQUIRED so the &kernel<T> expressions resolve to the CALLING TU's host stubs — otherwise the linker merges this function across TUs and we set the attribute on one TU's stubs while the launch goes through a different TU's.
     *
     */
    /**
     * Library-safe MaxDynamicSharedMemorySize registration for every algorithm kernel: returns the first cudaFuncSetAttribute/fit-check error and names it (no resources to release)
     *
     * @param failed_op (optional) receives a static string naming the failed operation
     * @return cudaSuccess or the first error
     */
    template <typename T>
    __host__ __forceinline__
    cudaError_t init_grid_kernel_attrs_checked(const char **failed_op = nullptr){
        // enable opt-in dynamic shared memory for every algorithm kernel
        // Gate registration on the DEVICE opt-in max (not the codegen target):
        // grid_check_dynamic_shared_memory_bytes and the bench's
        // grid_kernel_fits_device both use the device cap, so registering only up
        // to the smaller GRID_CUDA_TARGET_SHARED_MEM_BYTES left kernels in
        // (target, device-max] checkable+launchable but UNregistered -> launching
        // them failed with cudaErrorInvalidValue (e.g. the floating idsva_so
        // body-frame diagnostic ~101 KB on g1). Keying on the device max keeps
        // registration, the fit-check, and the launch-skip in lockstep.
        size_t _grid_smem_max = 0; { cudaError_t _e = GRID_CUDA_CALL(grid_get_max_dynamic_shared_memory_bytes(&_grid_smem_max)); if (_e != cudaSuccess) { return grid_fail(failed_op, "grid_get_max_dynamic_shared_memory_bytes(&_grid_smem_max))", _e); } }
        if (INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>() <= _grid_smem_max) {
            { cudaError_t _e = GRID_CUDA_CALL(grid_check_dynamic_shared_memory_bytes("inverse_dynamics", INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>())); if (_e != cudaSuccess) { return grid_fail(failed_op, "grid_check_dynamic_shared_memory_bytes(inverse_dynamics)", _e); } }
            auto _grid_kern_alias_0 = static_cast<void (*)(T *, const T *, const int, const T *, T *, const robotModel<T> *, const T, const int)>(&inverse_dynamics_kernel<T>);
            { cudaError_t _e = GRID_CUDA_CALL(cudaFuncSetAttribute(_grid_kern_alias_0, cudaFuncAttributeMaxDynamicSharedMemorySize, INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>())); if (_e != cudaSuccess) { return grid_fail(failed_op, "cudaFuncSetAttribute(_grid_kern_alias_0)", _e); } }
            auto _grid_kern_alias_1 = static_cast<void (*)(T *, const T *, const int, T *, const robotModel<T> *, const T, const int)>(&inverse_dynamics_kernel<T>);
            { cudaError_t _e = GRID_CUDA_CALL(cudaFuncSetAttribute(_grid_kern_alias_1, cudaFuncAttributeMaxDynamicSharedMemorySize, INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>())); if (_e != cudaSuccess) { return grid_fail(failed_op, "cudaFuncSetAttribute(_grid_kern_alias_1)", _e); } }
            auto _grid_kern_alias_2 = static_cast<void (*)(T *, const T *, const int, const T *, T *, const robotModel<T> *, const T, const int)>(&inverse_dynamics_kernel_single_timing<T>);
            { cudaError_t _e = GRID_CUDA_CALL(cudaFuncSetAttribute(_grid_kern_alias_2, cudaFuncAttributeMaxDynamicSharedMemorySize, INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>())); if (_e != cudaSuccess) { return grid_fail(failed_op, "cudaFuncSetAttribute(_grid_kern_alias_2)", _e); } }
            auto _grid_kern_alias_3 = static_cast<void (*)(T *, const T *, const int, T *, const robotModel<T> *, const T, const int)>(&inverse_dynamics_kernel_single_timing<T>);
            { cudaError_t _e = GRID_CUDA_CALL(cudaFuncSetAttribute(_grid_kern_alias_3, cudaFuncAttributeMaxDynamicSharedMemorySize, INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>())); if (_e != cudaSuccess) { return grid_fail(failed_op, "cudaFuncSetAttribute(_grid_kern_alias_3)", _e); } }
        }
        // inverse_dynamics: baked launch_cfg tier TIER_MINIMAL != default — the launchers
        // instantiate THAT kernel, so it needs its own dynamic-smem opt-in.
        if (INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>() <= _grid_smem_max) {
            { cudaError_t _e = GRID_CUDA_CALL(grid_check_dynamic_shared_memory_bytes("inverse_dynamics@TIER_MINIMAL", INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>())); if (_e != cudaSuccess) { return grid_fail(failed_op, "grid_check_dynamic_shared_memory_bytes(inverse_dynamics@TIER_MINIMAL)", _e); } }
            auto _grid_kern_alias_4 = static_cast<void (*)(T *, const T *, const int, const T *, T *, const robotModel<T> *, const T, const int)>(&inverse_dynamics_kernel<T, TIER_MINIMAL>);
            { cudaError_t _e = GRID_CUDA_CALL(cudaFuncSetAttribute(_grid_kern_alias_4, cudaFuncAttributeMaxDynamicSharedMemorySize, INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>())); if (_e != cudaSuccess) { return grid_fail(failed_op, "cudaFuncSetAttribute(_grid_kern_alias_4)", _e); } }
            auto _grid_kern_alias_5 = static_cast<void (*)(T *, const T *, const int, T *, const robotModel<T> *, const T, const int)>(&inverse_dynamics_kernel<T, TIER_MINIMAL>);
            { cudaError_t _e = GRID_CUDA_CALL(cudaFuncSetAttribute(_grid_kern_alias_5, cudaFuncAttributeMaxDynamicSharedMemorySize, INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>())); if (_e != cudaSuccess) { return grid_fail(failed_op, "cudaFuncSetAttribute(_grid_kern_alias_5)", _e); } }
            auto _grid_kern_alias_6 = static_cast<void (*)(T *, const T *, const int, const T *, T *, const robotModel<T> *, const T, const int)>(&inverse_dynamics_kernel_single_timing<T, TIER_MINIMAL>);
            { cudaError_t _e = GRID_CUDA_CALL(cudaFuncSetAttribute(_grid_kern_alias_6, cudaFuncAttributeMaxDynamicSharedMemorySize, INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>())); if (_e != cudaSuccess) { return grid_fail(failed_op, "cudaFuncSetAttribute(_grid_kern_alias_6)", _e); } }
            auto _grid_kern_alias_7 = static_cast<void (*)(T *, const T *, const int, T *, const robotModel<T> *, const T, const int)>(&inverse_dynamics_kernel_single_timing<T, TIER_MINIMAL>);
            { cudaError_t _e = GRID_CUDA_CALL(cudaFuncSetAttribute(_grid_kern_alias_7, cudaFuncAttributeMaxDynamicSharedMemorySize, INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>())); if (_e != cudaSuccess) { return grid_fail(failed_op, "cudaFuncSetAttribute(_grid_kern_alias_7)", _e); } }
        }
        if (MINV_DYNAMIC_SHARED_MEM_BYTES<T>() <= _grid_smem_max) {
            { cudaError_t _e = GRID_CUDA_CALL(grid_check_dynamic_shared_memory_bytes("minv", MINV_DYNAMIC_SHARED_MEM_BYTES<T>())); if (_e != cudaSuccess) { return grid_fail(failed_op, "grid_check_dynamic_shared_memory_bytes(minv)", _e); } }
            auto _grid_kern_alias_8 = static_cast<void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)>(&minv_kernel<T>);
            { cudaError_t _e = GRID_CUDA_CALL(cudaFuncSetAttribute(_grid_kern_alias_8, cudaFuncAttributeMaxDynamicSharedMemorySize, MINV_DYNAMIC_SHARED_MEM_BYTES<T>())); if (_e != cudaSuccess) { return grid_fail(failed_op, "cudaFuncSetAttribute(_grid_kern_alias_8)", _e); } }
            auto _grid_kern_alias_9 = static_cast<void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)>(&minv_kernel_single_timing<T>);
            { cudaError_t _e = GRID_CUDA_CALL(cudaFuncSetAttribute(_grid_kern_alias_9, cudaFuncAttributeMaxDynamicSharedMemorySize, MINV_DYNAMIC_SHARED_MEM_BYTES<T>())); if (_e != cudaSuccess) { return grid_fail(failed_op, "cudaFuncSetAttribute(_grid_kern_alias_9)", _e); } }
        }
        // minv: baked launch_cfg tier TIER_LITE != default — the launchers
        // instantiate THAT kernel, so it needs its own dynamic-smem opt-in.
        if (MINV_DYNAMIC_SHARED_MEM_BYTES<T, TIER_LITE>() <= _grid_smem_max) {
            { cudaError_t _e = GRID_CUDA_CALL(grid_check_dynamic_shared_memory_bytes("minv@TIER_LITE", MINV_DYNAMIC_SHARED_MEM_BYTES<T, TIER_LITE>())); if (_e != cudaSuccess) { return grid_fail(failed_op, "grid_check_dynamic_shared_memory_bytes(minv@TIER_LITE)", _e); } }
            auto _grid_kern_alias_10 = static_cast<void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)>(&minv_kernel<T, TIER_LITE>);
            { cudaError_t _e = GRID_CUDA_CALL(cudaFuncSetAttribute(_grid_kern_alias_10, cudaFuncAttributeMaxDynamicSharedMemorySize, MINV_DYNAMIC_SHARED_MEM_BYTES<T, TIER_LITE>())); if (_e != cudaSuccess) { return grid_fail(failed_op, "cudaFuncSetAttribute(_grid_kern_alias_10)", _e); } }
            auto _grid_kern_alias_11 = static_cast<void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)>(&minv_kernel_single_timing<T, TIER_LITE>);
            { cudaError_t _e = GRID_CUDA_CALL(cudaFuncSetAttribute(_grid_kern_alias_11, cudaFuncAttributeMaxDynamicSharedMemorySize, MINV_DYNAMIC_SHARED_MEM_BYTES<T, TIER_LITE>())); if (_e != cudaSuccess) { return grid_fail(failed_op, "cudaFuncSetAttribute(_grid_kern_alias_11)", _e); } }
        }
        if (FORWARD_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>() <= _grid_smem_max) {
            { cudaError_t _e = GRID_CUDA_CALL(grid_check_dynamic_shared_memory_bytes("forward_dynamics", FORWARD_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>())); if (_e != cudaSuccess) { return grid_fail(failed_op, "grid_check_dynamic_shared_memory_bytes(forward_dynamics)", _e); } }
            auto _grid_kern_alias_12 = static_cast<void (*)(T *, unsigned char *, const T *, const int, T *, const robotModel<T> *, const T, const int)>(&forward_dynamics_kernel<T>);
            { cudaError_t _e = GRID_CUDA_CALL(cudaFuncSetAttribute(_grid_kern_alias_12, cudaFuncAttributeMaxDynamicSharedMemorySize, FORWARD_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>())); if (_e != cudaSuccess) { return grid_fail(failed_op, "cudaFuncSetAttribute(_grid_kern_alias_12)", _e); } }
            auto _grid_kern_alias_13 = static_cast<void (*)(T *, unsigned char *, const T *, const int, T *, const robotModel<T> *, const T, const int)>(&forward_dynamics_kernel_single_timing<T>);
            { cudaError_t _e = GRID_CUDA_CALL(cudaFuncSetAttribute(_grid_kern_alias_13, cudaFuncAttributeMaxDynamicSharedMemorySize, FORWARD_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>())); if (_e != cudaSuccess) { return grid_fail(failed_op, "cudaFuncSetAttribute(_grid_kern_alias_13)", _e); } }
        }
        if (END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_BYTES<T>() <= _grid_smem_max) {
            { cudaError_t _e = GRID_CUDA_CALL(grid_check_dynamic_shared_memory_bytes("end_effector_pose", END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_BYTES<T>())); if (_e != cudaSuccess) { return grid_fail(failed_op, "grid_check_dynamic_shared_memory_bytes(end_effector_pose)", _e); } }
            auto _grid_kern_alias_14 = static_cast<void (*)(T *, const T *, const int, const robotModel<T> *, const int)>(&end_effector_pose_kernel<T>);
            { cudaError_t _e = GRID_CUDA_CALL(cudaFuncSetAttribute(_grid_kern_alias_14, cudaFuncAttributeMaxDynamicSharedMemorySize, END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_BYTES<T>())); if (_e != cudaSuccess) { return grid_fail(failed_op, "cudaFuncSetAttribute(_grid_kern_alias_14)", _e); } }
            auto _grid_kern_alias_15 = static_cast<void (*)(T *, const T *, const int, const robotModel<T> *, const int)>(&end_effector_pose_kernel_single_timing<T>);
            { cudaError_t _e = GRID_CUDA_CALL(cudaFuncSetAttribute(_grid_kern_alias_15, cudaFuncAttributeMaxDynamicSharedMemorySize, END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_BYTES<T>())); if (_e != cudaSuccess) { return grid_fail(failed_op, "cudaFuncSetAttribute(_grid_kern_alias_15)", _e); } }
        }
        // end_effector_pose: baked launch_cfg tier TIER_LITE != default — the launchers
        // instantiate THAT kernel, so it needs its own dynamic-smem opt-in.
        if (END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_BYTES<T>() <= _grid_smem_max) {
            { cudaError_t _e = GRID_CUDA_CALL(grid_check_dynamic_shared_memory_bytes("end_effector_pose@TIER_LITE", END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_BYTES<T>())); if (_e != cudaSuccess) { return grid_fail(failed_op, "grid_check_dynamic_shared_memory_bytes(end_effector_pose@TIER_LITE)", _e); } }
            auto _grid_kern_alias_16 = static_cast<void (*)(T *, const T *, const int, const robotModel<T> *, const int)>(&end_effector_pose_kernel<T, TIER_LITE>);
            { cudaError_t _e = GRID_CUDA_CALL(cudaFuncSetAttribute(_grid_kern_alias_16, cudaFuncAttributeMaxDynamicSharedMemorySize, END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_BYTES<T>())); if (_e != cudaSuccess) { return grid_fail(failed_op, "cudaFuncSetAttribute(_grid_kern_alias_16)", _e); } }
            auto _grid_kern_alias_17 = static_cast<void (*)(T *, const T *, const int, const robotModel<T> *, const int)>(&end_effector_pose_kernel_single_timing<T, TIER_LITE>);
            { cudaError_t _e = GRID_CUDA_CALL(cudaFuncSetAttribute(_grid_kern_alias_17, cudaFuncAttributeMaxDynamicSharedMemorySize, END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_BYTES<T>())); if (_e != cudaSuccess) { return grid_fail(failed_op, "cudaFuncSetAttribute(_grid_kern_alias_17)", _e); } }
        }
        if (END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>() <= _grid_smem_max) {
            { cudaError_t _e = GRID_CUDA_CALL(grid_check_dynamic_shared_memory_bytes("end_effector_pose_gradient", END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>())); if (_e != cudaSuccess) { return grid_fail(failed_op, "grid_check_dynamic_shared_memory_bytes(end_effector_pose_gradient)", _e); } }
            auto _grid_kern_alias_18 = static_cast<void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)>(&end_effector_pose_gradient_kernel<T>);
            { cudaError_t _e = GRID_CUDA_CALL(cudaFuncSetAttribute(_grid_kern_alias_18, cudaFuncAttributeMaxDynamicSharedMemorySize, END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>())); if (_e != cudaSuccess) { return grid_fail(failed_op, "cudaFuncSetAttribute(_grid_kern_alias_18)", _e); } }
            auto _grid_kern_alias_19 = static_cast<void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)>(&end_effector_pose_gradient_kernel_single_timing<T>);
            { cudaError_t _e = GRID_CUDA_CALL(cudaFuncSetAttribute(_grid_kern_alias_19, cudaFuncAttributeMaxDynamicSharedMemorySize, END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>())); if (_e != cudaSuccess) { return grid_fail(failed_op, "cudaFuncSetAttribute(_grid_kern_alias_19)", _e); } }
        }
        // end_effector_pose_gradient: baked launch_cfg tier TIER_MINIMAL != default — the launchers
        // instantiate THAT kernel, so it needs its own dynamic-smem opt-in.
        if (END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, TIER_MINIMAL>() <= _grid_smem_max) {
            { cudaError_t _e = GRID_CUDA_CALL(grid_check_dynamic_shared_memory_bytes("end_effector_pose_gradient@TIER_MINIMAL", END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, TIER_MINIMAL>())); if (_e != cudaSuccess) { return grid_fail(failed_op, "grid_check_dynamic_shared_memory_bytes(end_effector_pose_gradient@TIER_MINIMAL)", _e); } }
            auto _grid_kern_alias_20 = static_cast<void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)>(&end_effector_pose_gradient_kernel<T, TIER_MINIMAL>);
            { cudaError_t _e = GRID_CUDA_CALL(cudaFuncSetAttribute(_grid_kern_alias_20, cudaFuncAttributeMaxDynamicSharedMemorySize, END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, TIER_MINIMAL>())); if (_e != cudaSuccess) { return grid_fail(failed_op, "cudaFuncSetAttribute(_grid_kern_alias_20)", _e); } }
            auto _grid_kern_alias_21 = static_cast<void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)>(&end_effector_pose_gradient_kernel_single_timing<T, TIER_MINIMAL>);
            { cudaError_t _e = GRID_CUDA_CALL(cudaFuncSetAttribute(_grid_kern_alias_21, cudaFuncAttributeMaxDynamicSharedMemorySize, END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, TIER_MINIMAL>())); if (_e != cudaSuccess) { return grid_fail(failed_op, "cudaFuncSetAttribute(_grid_kern_alias_21)", _e); } }
        }
        if (INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>() <= _grid_smem_max) {
            { cudaError_t _e = GRID_CUDA_CALL(grid_check_dynamic_shared_memory_bytes("inverse_dynamics_gradient", INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>())); if (_e != cudaSuccess) { return grid_fail(failed_op, "grid_check_dynamic_shared_memory_bytes(inverse_dynamics_gradient)", _e); } }
            auto _grid_kern_alias_22 = static_cast<void (*)(T *, unsigned char *, const T *, const int, const T *, T *, const robotModel<T> *, const T, const int)>(&inverse_dynamics_gradient_kernel<T>);
            { cudaError_t _e = GRID_CUDA_CALL(cudaFuncSetAttribute(_grid_kern_alias_22, cudaFuncAttributeMaxDynamicSharedMemorySize, INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>())); if (_e != cudaSuccess) { return grid_fail(failed_op, "cudaFuncSetAttribute(_grid_kern_alias_22)", _e); } }
            auto _grid_kern_alias_23 = static_cast<void (*)(T *, unsigned char *, const T *, const int, T *, const robotModel<T> *, const T, const int)>(&inverse_dynamics_gradient_kernel<T>);
            { cudaError_t _e = GRID_CUDA_CALL(cudaFuncSetAttribute(_grid_kern_alias_23, cudaFuncAttributeMaxDynamicSharedMemorySize, INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>())); if (_e != cudaSuccess) { return grid_fail(failed_op, "cudaFuncSetAttribute(_grid_kern_alias_23)", _e); } }
            auto _grid_kern_alias_24 = static_cast<void (*)(T *, unsigned char *, const T *, const int, const T *, T *, const robotModel<T> *, const T, const int)>(&inverse_dynamics_gradient_kernel_single_timing<T>);
            { cudaError_t _e = GRID_CUDA_CALL(cudaFuncSetAttribute(_grid_kern_alias_24, cudaFuncAttributeMaxDynamicSharedMemorySize, INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>())); if (_e != cudaSuccess) { return grid_fail(failed_op, "cudaFuncSetAttribute(_grid_kern_alias_24)", _e); } }
            auto _grid_kern_alias_25 = static_cast<void (*)(T *, unsigned char *, const T *, const int, T *, const robotModel<T> *, const T, const int)>(&inverse_dynamics_gradient_kernel_single_timing<T>);
            { cudaError_t _e = GRID_CUDA_CALL(cudaFuncSetAttribute(_grid_kern_alias_25, cudaFuncAttributeMaxDynamicSharedMemorySize, INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>())); if (_e != cudaSuccess) { return grid_fail(failed_op, "cudaFuncSetAttribute(_grid_kern_alias_25)", _e); } }
        }
        if (FORWARD_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>() <= _grid_smem_max) {
            { cudaError_t _e = GRID_CUDA_CALL(grid_check_dynamic_shared_memory_bytes("forward_dynamics_gradient", FORWARD_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>())); if (_e != cudaSuccess) { return grid_fail(failed_op, "grid_check_dynamic_shared_memory_bytes(forward_dynamics_gradient)", _e); } }
            auto _grid_kern_alias_26 = static_cast<void (*)(T *, unsigned char *, const T *, const int, const T *, const T *, T *, const robotModel<T> *, const T, const int)>(&forward_dynamics_gradient_kernel<T>);
            { cudaError_t _e = GRID_CUDA_CALL(cudaFuncSetAttribute(_grid_kern_alias_26, cudaFuncAttributeMaxDynamicSharedMemorySize, FORWARD_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>())); if (_e != cudaSuccess) { return grid_fail(failed_op, "cudaFuncSetAttribute(_grid_kern_alias_26)", _e); } }
            auto _grid_kern_alias_27 = static_cast<void (*)(T *, unsigned char *, const T *, const int, T *, const robotModel<T> *, const T, const int)>(&forward_dynamics_gradient_kernel<T>);
            { cudaError_t _e = GRID_CUDA_CALL(cudaFuncSetAttribute(_grid_kern_alias_27, cudaFuncAttributeMaxDynamicSharedMemorySize, FORWARD_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>())); if (_e != cudaSuccess) { return grid_fail(failed_op, "cudaFuncSetAttribute(_grid_kern_alias_27)", _e); } }
            auto _grid_kern_alias_28 = static_cast<void (*)(T *, unsigned char *, const T *, const int, const T *, const T *, T *, const robotModel<T> *, const T, const int)>(&forward_dynamics_gradient_kernel_single_timing<T>);
            { cudaError_t _e = GRID_CUDA_CALL(cudaFuncSetAttribute(_grid_kern_alias_28, cudaFuncAttributeMaxDynamicSharedMemorySize, FORWARD_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>())); if (_e != cudaSuccess) { return grid_fail(failed_op, "cudaFuncSetAttribute(_grid_kern_alias_28)", _e); } }
            auto _grid_kern_alias_29 = static_cast<void (*)(T *, unsigned char *, const T *, const int, T *, const robotModel<T> *, const T, const int)>(&forward_dynamics_gradient_kernel_single_timing<T>);
            { cudaError_t _e = GRID_CUDA_CALL(cudaFuncSetAttribute(_grid_kern_alias_29, cudaFuncAttributeMaxDynamicSharedMemorySize, FORWARD_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>())); if (_e != cudaSuccess) { return grid_fail(failed_op, "cudaFuncSetAttribute(_grid_kern_alias_29)", _e); } }
        }
        return cudaSuccess;
    }

    template <typename T>
    __host__ __forceinline__
    void init_grid_kernel_attrs(){
        const char *op = nullptr;
        cudaError_t e = init_grid_kernel_attrs_checked<T>(&op);  // sequenced BEFORE reading op
        grid_legacy_check(e, op, __FILE__, __LINE__);
    }

    /**
     * Set MaxDynamicSharedMemorySize for the inverse_dynamics kernel(s) only (callable from any TU; idempotent). Split-compile entry point: registers just this algo so a solo TU instantiates only its kernel.
     *
     */
    template <typename T>
    __host__ __forceinline__
    void init_grid_kernel_attr_inverse_dynamics(){
        size_t _grid_smem_max = 0; gpuErrchk(grid_get_max_dynamic_shared_memory_bytes(&_grid_smem_max));
        if (INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>() <= _grid_smem_max) {
            gpuErrchk(grid_check_dynamic_shared_memory_bytes("inverse_dynamics", INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>()));
            auto _grid_kern_alias_0 = static_cast<void (*)(T *, const T *, const int, const T *, T *, const robotModel<T> *, const T, const int)>(&inverse_dynamics_kernel<T>);
            gpuErrchk(cudaFuncSetAttribute(_grid_kern_alias_0, cudaFuncAttributeMaxDynamicSharedMemorySize, INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>()));
            auto _grid_kern_alias_1 = static_cast<void (*)(T *, const T *, const int, T *, const robotModel<T> *, const T, const int)>(&inverse_dynamics_kernel<T>);
            gpuErrchk(cudaFuncSetAttribute(_grid_kern_alias_1, cudaFuncAttributeMaxDynamicSharedMemorySize, INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>()));
            auto _grid_kern_alias_2 = static_cast<void (*)(T *, const T *, const int, const T *, T *, const robotModel<T> *, const T, const int)>(&inverse_dynamics_kernel_single_timing<T>);
            gpuErrchk(cudaFuncSetAttribute(_grid_kern_alias_2, cudaFuncAttributeMaxDynamicSharedMemorySize, INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>()));
            auto _grid_kern_alias_3 = static_cast<void (*)(T *, const T *, const int, T *, const robotModel<T> *, const T, const int)>(&inverse_dynamics_kernel_single_timing<T>);
            gpuErrchk(cudaFuncSetAttribute(_grid_kern_alias_3, cudaFuncAttributeMaxDynamicSharedMemorySize, INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>()));
        }
        // inverse_dynamics: baked launch_cfg tier TIER_MINIMAL != default — the launchers
        // instantiate THAT kernel, so it needs its own dynamic-smem opt-in.
        if (INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>() <= _grid_smem_max) {
            gpuErrchk(grid_check_dynamic_shared_memory_bytes("inverse_dynamics@TIER_MINIMAL", INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>()));
            auto _grid_kern_alias_4 = static_cast<void (*)(T *, const T *, const int, const T *, T *, const robotModel<T> *, const T, const int)>(&inverse_dynamics_kernel<T, TIER_MINIMAL>);
            gpuErrchk(cudaFuncSetAttribute(_grid_kern_alias_4, cudaFuncAttributeMaxDynamicSharedMemorySize, INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>()));
            auto _grid_kern_alias_5 = static_cast<void (*)(T *, const T *, const int, T *, const robotModel<T> *, const T, const int)>(&inverse_dynamics_kernel<T, TIER_MINIMAL>);
            gpuErrchk(cudaFuncSetAttribute(_grid_kern_alias_5, cudaFuncAttributeMaxDynamicSharedMemorySize, INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>()));
            auto _grid_kern_alias_6 = static_cast<void (*)(T *, const T *, const int, const T *, T *, const robotModel<T> *, const T, const int)>(&inverse_dynamics_kernel_single_timing<T, TIER_MINIMAL>);
            gpuErrchk(cudaFuncSetAttribute(_grid_kern_alias_6, cudaFuncAttributeMaxDynamicSharedMemorySize, INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>()));
            auto _grid_kern_alias_7 = static_cast<void (*)(T *, const T *, const int, T *, const robotModel<T> *, const T, const int)>(&inverse_dynamics_kernel_single_timing<T, TIER_MINIMAL>);
            gpuErrchk(cudaFuncSetAttribute(_grid_kern_alias_7, cudaFuncAttributeMaxDynamicSharedMemorySize, INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>()));
        }
    }

    /**
     * Set MaxDynamicSharedMemorySize for the minv kernel(s) only (callable from any TU; idempotent). Split-compile entry point: registers just this algo so a solo TU instantiates only its kernel.
     *
     */
    template <typename T>
    __host__ __forceinline__
    void init_grid_kernel_attr_minv(){
        size_t _grid_smem_max = 0; gpuErrchk(grid_get_max_dynamic_shared_memory_bytes(&_grid_smem_max));
        if (MINV_DYNAMIC_SHARED_MEM_BYTES<T>() <= _grid_smem_max) {
            gpuErrchk(grid_check_dynamic_shared_memory_bytes("minv", MINV_DYNAMIC_SHARED_MEM_BYTES<T>()));
            auto _grid_kern_alias_0 = static_cast<void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)>(&minv_kernel<T>);
            gpuErrchk(cudaFuncSetAttribute(_grid_kern_alias_0, cudaFuncAttributeMaxDynamicSharedMemorySize, MINV_DYNAMIC_SHARED_MEM_BYTES<T>()));
            auto _grid_kern_alias_1 = static_cast<void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)>(&minv_kernel_single_timing<T>);
            gpuErrchk(cudaFuncSetAttribute(_grid_kern_alias_1, cudaFuncAttributeMaxDynamicSharedMemorySize, MINV_DYNAMIC_SHARED_MEM_BYTES<T>()));
        }
        // minv: baked launch_cfg tier TIER_LITE != default — the launchers
        // instantiate THAT kernel, so it needs its own dynamic-smem opt-in.
        if (MINV_DYNAMIC_SHARED_MEM_BYTES<T, TIER_LITE>() <= _grid_smem_max) {
            gpuErrchk(grid_check_dynamic_shared_memory_bytes("minv@TIER_LITE", MINV_DYNAMIC_SHARED_MEM_BYTES<T, TIER_LITE>()));
            auto _grid_kern_alias_2 = static_cast<void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)>(&minv_kernel<T, TIER_LITE>);
            gpuErrchk(cudaFuncSetAttribute(_grid_kern_alias_2, cudaFuncAttributeMaxDynamicSharedMemorySize, MINV_DYNAMIC_SHARED_MEM_BYTES<T, TIER_LITE>()));
            auto _grid_kern_alias_3 = static_cast<void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)>(&minv_kernel_single_timing<T, TIER_LITE>);
            gpuErrchk(cudaFuncSetAttribute(_grid_kern_alias_3, cudaFuncAttributeMaxDynamicSharedMemorySize, MINV_DYNAMIC_SHARED_MEM_BYTES<T, TIER_LITE>()));
        }
    }

    /**
     * Set MaxDynamicSharedMemorySize for the forward_dynamics kernel(s) only (callable from any TU; idempotent). Split-compile entry point: registers just this algo so a solo TU instantiates only its kernel.
     *
     */
    template <typename T>
    __host__ __forceinline__
    void init_grid_kernel_attr_forward_dynamics(){
        size_t _grid_smem_max = 0; gpuErrchk(grid_get_max_dynamic_shared_memory_bytes(&_grid_smem_max));
        if (FORWARD_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>() <= _grid_smem_max) {
            gpuErrchk(grid_check_dynamic_shared_memory_bytes("forward_dynamics", FORWARD_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>()));
            auto _grid_kern_alias_0 = static_cast<void (*)(T *, unsigned char *, const T *, const int, T *, const robotModel<T> *, const T, const int)>(&forward_dynamics_kernel<T>);
            gpuErrchk(cudaFuncSetAttribute(_grid_kern_alias_0, cudaFuncAttributeMaxDynamicSharedMemorySize, FORWARD_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>()));
            auto _grid_kern_alias_1 = static_cast<void (*)(T *, unsigned char *, const T *, const int, T *, const robotModel<T> *, const T, const int)>(&forward_dynamics_kernel_single_timing<T>);
            gpuErrchk(cudaFuncSetAttribute(_grid_kern_alias_1, cudaFuncAttributeMaxDynamicSharedMemorySize, FORWARD_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>()));
        }
    }

    /**
     * Set MaxDynamicSharedMemorySize for the end_effector_pose kernel(s) only (callable from any TU; idempotent). Split-compile entry point: registers just this algo so a solo TU instantiates only its kernel.
     *
     */
    template <typename T>
    __host__ __forceinline__
    void init_grid_kernel_attr_end_effector_pose(){
        size_t _grid_smem_max = 0; gpuErrchk(grid_get_max_dynamic_shared_memory_bytes(&_grid_smem_max));
        if (END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_BYTES<T>() <= _grid_smem_max) {
            gpuErrchk(grid_check_dynamic_shared_memory_bytes("end_effector_pose", END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_BYTES<T>()));
            auto _grid_kern_alias_0 = static_cast<void (*)(T *, const T *, const int, const robotModel<T> *, const int)>(&end_effector_pose_kernel<T>);
            gpuErrchk(cudaFuncSetAttribute(_grid_kern_alias_0, cudaFuncAttributeMaxDynamicSharedMemorySize, END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_BYTES<T>()));
            auto _grid_kern_alias_1 = static_cast<void (*)(T *, const T *, const int, const robotModel<T> *, const int)>(&end_effector_pose_kernel_single_timing<T>);
            gpuErrchk(cudaFuncSetAttribute(_grid_kern_alias_1, cudaFuncAttributeMaxDynamicSharedMemorySize, END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_BYTES<T>()));
        }
        // end_effector_pose: baked launch_cfg tier TIER_LITE != default — the launchers
        // instantiate THAT kernel, so it needs its own dynamic-smem opt-in.
        if (END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_BYTES<T>() <= _grid_smem_max) {
            gpuErrchk(grid_check_dynamic_shared_memory_bytes("end_effector_pose@TIER_LITE", END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_BYTES<T>()));
            auto _grid_kern_alias_2 = static_cast<void (*)(T *, const T *, const int, const robotModel<T> *, const int)>(&end_effector_pose_kernel<T, TIER_LITE>);
            gpuErrchk(cudaFuncSetAttribute(_grid_kern_alias_2, cudaFuncAttributeMaxDynamicSharedMemorySize, END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_BYTES<T>()));
            auto _grid_kern_alias_3 = static_cast<void (*)(T *, const T *, const int, const robotModel<T> *, const int)>(&end_effector_pose_kernel_single_timing<T, TIER_LITE>);
            gpuErrchk(cudaFuncSetAttribute(_grid_kern_alias_3, cudaFuncAttributeMaxDynamicSharedMemorySize, END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_BYTES<T>()));
        }
    }

    /**
     * Set MaxDynamicSharedMemorySize for the end_effector_pose_gradient kernel(s) only (callable from any TU; idempotent). Split-compile entry point: registers just this algo so a solo TU instantiates only its kernel.
     *
     */
    template <typename T>
    __host__ __forceinline__
    void init_grid_kernel_attr_end_effector_pose_gradient(){
        size_t _grid_smem_max = 0; gpuErrchk(grid_get_max_dynamic_shared_memory_bytes(&_grid_smem_max));
        if (END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>() <= _grid_smem_max) {
            gpuErrchk(grid_check_dynamic_shared_memory_bytes("end_effector_pose_gradient", END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>()));
            auto _grid_kern_alias_0 = static_cast<void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)>(&end_effector_pose_gradient_kernel<T>);
            gpuErrchk(cudaFuncSetAttribute(_grid_kern_alias_0, cudaFuncAttributeMaxDynamicSharedMemorySize, END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>()));
            auto _grid_kern_alias_1 = static_cast<void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)>(&end_effector_pose_gradient_kernel_single_timing<T>);
            gpuErrchk(cudaFuncSetAttribute(_grid_kern_alias_1, cudaFuncAttributeMaxDynamicSharedMemorySize, END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>()));
        }
        // end_effector_pose_gradient: baked launch_cfg tier TIER_MINIMAL != default — the launchers
        // instantiate THAT kernel, so it needs its own dynamic-smem opt-in.
        if (END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, TIER_MINIMAL>() <= _grid_smem_max) {
            gpuErrchk(grid_check_dynamic_shared_memory_bytes("end_effector_pose_gradient@TIER_MINIMAL", END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, TIER_MINIMAL>()));
            auto _grid_kern_alias_2 = static_cast<void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)>(&end_effector_pose_gradient_kernel<T, TIER_MINIMAL>);
            gpuErrchk(cudaFuncSetAttribute(_grid_kern_alias_2, cudaFuncAttributeMaxDynamicSharedMemorySize, END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, TIER_MINIMAL>()));
            auto _grid_kern_alias_3 = static_cast<void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)>(&end_effector_pose_gradient_kernel_single_timing<T, TIER_MINIMAL>);
            gpuErrchk(cudaFuncSetAttribute(_grid_kern_alias_3, cudaFuncAttributeMaxDynamicSharedMemorySize, END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, TIER_MINIMAL>()));
        }
    }

    /**
     * Set MaxDynamicSharedMemorySize for the inverse_dynamics_gradient kernel(s) only (callable from any TU; idempotent). Split-compile entry point: registers just this algo so a solo TU instantiates only its kernel.
     *
     */
    template <typename T>
    __host__ __forceinline__
    void init_grid_kernel_attr_inverse_dynamics_gradient(){
        size_t _grid_smem_max = 0; gpuErrchk(grid_get_max_dynamic_shared_memory_bytes(&_grid_smem_max));
        if (INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>() <= _grid_smem_max) {
            gpuErrchk(grid_check_dynamic_shared_memory_bytes("inverse_dynamics_gradient", INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>()));
            auto _grid_kern_alias_0 = static_cast<void (*)(T *, unsigned char *, const T *, const int, const T *, T *, const robotModel<T> *, const T, const int)>(&inverse_dynamics_gradient_kernel<T>);
            gpuErrchk(cudaFuncSetAttribute(_grid_kern_alias_0, cudaFuncAttributeMaxDynamicSharedMemorySize, INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>()));
            auto _grid_kern_alias_1 = static_cast<void (*)(T *, unsigned char *, const T *, const int, T *, const robotModel<T> *, const T, const int)>(&inverse_dynamics_gradient_kernel<T>);
            gpuErrchk(cudaFuncSetAttribute(_grid_kern_alias_1, cudaFuncAttributeMaxDynamicSharedMemorySize, INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>()));
            auto _grid_kern_alias_2 = static_cast<void (*)(T *, unsigned char *, const T *, const int, const T *, T *, const robotModel<T> *, const T, const int)>(&inverse_dynamics_gradient_kernel_single_timing<T>);
            gpuErrchk(cudaFuncSetAttribute(_grid_kern_alias_2, cudaFuncAttributeMaxDynamicSharedMemorySize, INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>()));
            auto _grid_kern_alias_3 = static_cast<void (*)(T *, unsigned char *, const T *, const int, T *, const robotModel<T> *, const T, const int)>(&inverse_dynamics_gradient_kernel_single_timing<T>);
            gpuErrchk(cudaFuncSetAttribute(_grid_kern_alias_3, cudaFuncAttributeMaxDynamicSharedMemorySize, INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>()));
        }
    }

    /**
     * Set MaxDynamicSharedMemorySize for the forward_dynamics_gradient kernel(s) only (callable from any TU; idempotent). Split-compile entry point: registers just this algo so a solo TU instantiates only its kernel.
     *
     */
    template <typename T>
    __host__ __forceinline__
    void init_grid_kernel_attr_forward_dynamics_gradient(){
        size_t _grid_smem_max = 0; gpuErrchk(grid_get_max_dynamic_shared_memory_bytes(&_grid_smem_max));
        if (FORWARD_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>() <= _grid_smem_max) {
            gpuErrchk(grid_check_dynamic_shared_memory_bytes("forward_dynamics_gradient", FORWARD_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>()));
            auto _grid_kern_alias_0 = static_cast<void (*)(T *, unsigned char *, const T *, const int, const T *, const T *, T *, const robotModel<T> *, const T, const int)>(&forward_dynamics_gradient_kernel<T>);
            gpuErrchk(cudaFuncSetAttribute(_grid_kern_alias_0, cudaFuncAttributeMaxDynamicSharedMemorySize, FORWARD_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>()));
            auto _grid_kern_alias_1 = static_cast<void (*)(T *, unsigned char *, const T *, const int, T *, const robotModel<T> *, const T, const int)>(&forward_dynamics_gradient_kernel<T>);
            gpuErrchk(cudaFuncSetAttribute(_grid_kern_alias_1, cudaFuncAttributeMaxDynamicSharedMemorySize, FORWARD_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>()));
            auto _grid_kern_alias_2 = static_cast<void (*)(T *, unsigned char *, const T *, const int, const T *, const T *, T *, const robotModel<T> *, const T, const int)>(&forward_dynamics_gradient_kernel_single_timing<T>);
            gpuErrchk(cudaFuncSetAttribute(_grid_kern_alias_2, cudaFuncAttributeMaxDynamicSharedMemorySize, FORWARD_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>()));
            auto _grid_kern_alias_3 = static_cast<void (*)(T *, unsigned char *, const T *, const int, T *, const robotModel<T> *, const T, const int)>(&forward_dynamics_gradient_kernel_single_timing<T>);
            gpuErrchk(cudaFuncSetAttribute(_grid_kern_alias_3, cudaFuncAttributeMaxDynamicSharedMemorySize, FORWARD_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>()));
        }
    }

    /**
     * Allocates streams for host functions WITHOUT registering any kernel attributes (pair with an init_grid_kernel_attr_<algo> for split compiles).
     *
     * @return A pointer to the array of streams
     */
    /**
     * Library-safe stream allocation WITHOUT kernel-attribute registration: on failure every stream created by this attempt is destroyed and the failed operation named; *out published on success only
     *
     * @param out receives the stream array (nullptr on failure)
     * @param failed_op (optional)
     * @return cudaSuccess or the first error
     */
    template <typename T>
    __host__
    cudaError_t init_grid_streams_checked(cudaStream_t **out, const char **failed_op = nullptr){
        *out = nullptr;
        { cudaError_t _e = GRID_CUDA_CALL(cudaDeviceSynchronize()); if (_e != cudaSuccess) { return grid_fail(failed_op, "cudaDeviceSynchronize()", _e); } }
        // allocate streams
        cudaStream_t *streams = (cudaStream_t *)GRID_HOST_ALLOC(malloc(3*sizeof(cudaStream_t)));
        if (streams == nullptr) { return grid_fail(failed_op, "malloc(streams)", cudaErrorMemoryAllocation); }
        int priority, minPriority, maxPriority;
        { cudaError_t _e = GRID_CUDA_CALL(cudaDeviceGetStreamPriorityRange(&minPriority, &maxPriority)); if (_e != cudaSuccess) { free(streams); return grid_fail(failed_op, "cudaDeviceGetStreamPriorityRange()", _e); } }
        for(int i=0; i<3; i++){
            int adjusted_max = maxPriority - i; priority = adjusted_max > minPriority ? adjusted_max : minPriority;
            cudaError_t _e = GRID_CUDA_CALL(cudaStreamCreateWithPriority(&(streams[i]),cudaStreamDefault,priority));  // BLOCKING streams: every generated host wrapper copies inputs on streams[0] and launches kernels on streams[0]; a blocking stream keeps that ordered against the legacy default stream (`cudaStreamNonBlocking` here would let a kernel run BEFORE the H2D copy landed).
            if (_e != cudaSuccess) { for (int j = 0; j < i; j++) { GRID_CUDA_CALL(cudaStreamDestroy(streams[j])); } free(streams); return grid_fail(failed_op, "cudaStreamCreateWithPriority(streams[i])", _e); }
        }
        *out = streams;
        return cudaSuccess;
    }

    /**
     * Library-safe full init: kernel attributes then streams (see init_grid_kernel_attrs_checked / init_grid_streams_checked)
     *
     * @param out receives the stream array (nullptr on failure)
     * @param failed_op (optional)
     * @return cudaSuccess or the first error
     */
    template <typename T>
    __host__
    cudaError_t init_grid_checked(cudaStream_t **out, const char **failed_op = nullptr){
        *out = nullptr;
        { cudaError_t _e = init_grid_kernel_attrs_checked<T>(failed_op); if (_e != cudaSuccess) { return _e; } }
        return init_grid_streams_checked<T>(out, failed_op);
    }

    template <typename T>
    __host__
    cudaStream_t *init_grid_streams(){
        cudaStream_t *streams = nullptr; const char *op = nullptr;
        cudaError_t e = init_grid_streams_checked<T>(&streams, &op);  // sequenced BEFORE reading op
        grid_legacy_check(e, op, __FILE__, __LINE__);
        return streams;
    }

    /**
     * Sets MaxDynamicSharedMemorySize for every algorithm kernel and initializes streams for host functions
     *
     * @return A pointer to the array of streams
     */
    template <typename T>
    __host__
    cudaStream_t *init_grid(){
        cudaStream_t *streams = nullptr; const char *op = nullptr;
        cudaError_t e = init_grid_checked<T>(&streams, &op);  // sequenced BEFORE reading op
        grid_legacy_check(e, op, __FILE__, __LINE__);
        return streams;
    }

    /**
     * Library-safe teardown of streams, robotModel and gridData: every argument may be nullptr (no-op); cleanup continues past a failed free/destroy and the FIRST error is returned and named; never exit/abort/cudaDeviceReset
     *
     * @param streams allocated by init_grid[_checked] (or nullptr)
     * @param robotModel allocated by init_robotModel[_checked] (or nullptr)
     * @param data allocated by init_gridData[_checked] (or nullptr)
     * @param failed_op (optional)
     * @return cudaSuccess or the first error
     */
    template <typename T, gridDataKind KIND = GRID_DATA_ALL>
    __host__
    cudaError_t close_grid_checked(cudaStream_t *streams, robotModel<T> *d_robotModel, gridData<T, KIND> *hd_data, const char **failed_op = nullptr){
        cudaError_t first = cudaSuccess; const char *op = nullptr;
        { cudaError_t e = free_robotModel_checked<T>(d_robotModel, &op); if (e != cudaSuccess && first == cudaSuccess) { first = e; grid_fail(failed_op, op, e); } }
        if (hd_data != nullptr) {
            op = nullptr;
            { cudaError_t e = release_gridData_members<T, KIND>(hd_data, &op); if (e != cudaSuccess && first == cudaSuccess) { first = e; grid_fail(failed_op, op, e); } }
            // Phase 3a/b/c/e: end the L2 persisting window opened at init.
            { cudaError_t e = GRID_CUDA_CALL(grid_end_l2_persisting(0)); if (e != cudaSuccess && first == cudaSuccess) { first = e; grid_fail(failed_op, "grid_end_l2_persisting(0)", e); } }
            grid_device_pool_t *_pool = hd_data->pool;
            free(hd_data);
            // Device-pool mode: rewind the consumed slab (this arena's OWN pool, B1/K1) so a close/re-init cycle re-carves from the top.
            if (_pool != nullptr) { _pool->used = 0; }
        }
        if (streams != nullptr) {
            for(int i=0; i<3; i++){ cudaError_t e = GRID_CUDA_CALL(cudaStreamDestroy(streams[i])); if (e != cudaSuccess && first == cudaSuccess) { first = e; grid_fail(failed_op, "cudaStreamDestroy(streams[i])", e); } }
            free(streams);
        }
        return first;
    }

    /**
     * Frees the memory used by grid (legacy policy: exit on failure, or sticky first error under GRID_GPUERRCHK_NO_EXIT; prefer close_grid_checked in library code)
     *
     * @param streams allocated by init_grid
     * @param robotModel allocated by init_robotModel
     * @param data allocated by init_gridData
     */
    template <typename T, gridDataKind KIND = GRID_DATA_ALL>
    __host__
    void close_grid(cudaStream_t *streams, robotModel<T> *d_robotModel, gridData<T, KIND> *hd_data){
        const char *op = nullptr;
        cudaError_t e = close_grid_checked<T, KIND>(streams, d_robotModel, hd_data, &op);  // sequenced BEFORE reading op
        grid_legacy_check(e, op, __FILE__, __LINE__);
    }

}

/**
 * Plant namespace: cost / constraint / plant-step primitives composed over grid::
 *
 */
namespace grid_plant {
    /**
     * quadratic_state_cost: value = 1/2 * sum_i s_Q[i] * (s_x[i] - s_x_des[i])^2
     *
     * Notes:
     *   Block-cooperative: each thread accumulates its strided terms into s_scratch, then a serial reduction writes s_out[0].
     *   ACCUMULATE=false overwrites s_out[0]; ACCUMULATE=true ADDS into it (for summing cost terms into one scalar).
     *   s_scratch must hold at least 14 elements.
     *
     * @param s_out is the scalar cost output (s_out[0])
     * @param s_x is the current value (size NUM_POS + NUM_VEL = 14)
     * @param s_x_des is the desired/target value (size NUM_POS + NUM_VEL = 14)
     * @param s_Q is the diagonal weight vector (size NUM_POS + NUM_VEL = 14)
     * @param s_scratch is shared scratch of size >= 14
     */
    template <typename T, bool ACCUMULATE = false>
    __device__
    void quadratic_state_cost(T *s_out, const T *s_x, const T *s_x_des, const T *s_Q, T *s_scratch) {
        for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < 14; i += blockDim.x*blockDim.y){
            T r = s_x[i] - s_x_des[i];
            s_scratch[i] = static_cast<T>(0.5) * s_Q[i] * r * r;
        }
        __syncthreads();
        if(threadIdx.x == 0 && threadIdx.y == 0){
            T acc = static_cast<T>(0);
            for (int i = 0; i < 14; ++i) acc += s_scratch[i];
            if (ACCUMULATE) { s_out[0] += acc; } else { s_out[0] = acc; }
        }
    }

    /**
     * quadratic_state_cost_gradient: g[i] = s_Q[i] * (s_x[i] - s_x_des[i])
     *
     * Notes:
     *   ACCUMULATE=false overwrites s_grad; ACCUMULATE=true adds into it (for fusing into a packed [x;u] gradient).
     *
     * @param s_grad is the gradient output (size NUM_POS + NUM_VEL = 14)
     * @param s_x / s_x_des / s_Q as in the value function
     */
    template <typename T, bool ACCUMULATE = false, bool MUJOCO_OUTPUT = false>
    __device__
    void quadratic_state_cost_gradient(T *s_grad, const T *s_x, const T *s_x_des, const T *s_Q) {
        for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < 14; i += blockDim.x*blockDim.y){
            T g = s_Q[i] * (s_x[i] - s_x_des[i]);
            if (ACCUMULATE) { s_grad[i] += g; } else { s_grad[i] = g; }
        }
    }

    /**
     * quadratic_state_cost_hessian: Gauss-Newton hessian = diag(s_Q) (RATIFIED: GN outer product; for a quadratic cost this is exactly diag(W))
     *
     * Notes:
     *   Writes a dense column-major 14 x 14 matrix; off-diagonal entries are zero.
     *   ACCUMULATE=false overwrites; ACCUMULATE=true adds into the diagonal of an existing block.
     *
     * @param s_hess is the dense hessian output (size 14*14, column-major)
     * @param s_Q is the diagonal weight vector (size NUM_POS + NUM_VEL = 14)
     */
    template <typename T, bool ACCUMULATE = false, bool MUJOCO_OUTPUT = false>
    __device__
    void quadratic_state_cost_hessian(T *s_hess, const T *s_Q) {
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 196; ind += blockDim.x*blockDim.y){
            int row = ind % 14;
            int col = ind / 14;
            T h = (row == col) ? s_Q[row] : static_cast<T>(0);
            if (ACCUMULATE) { s_hess[ind] += h; } else { s_hess[ind] = h; }
        }
    }

    /**
     * quadratic_state_cost_value_grad_hess: fused value + gradient + GN-diag hessian in one pass
     *
     * Notes:
     *   Convenience fusion of the three functions above; same conventions and ACCUMULATE semantics for grad/hess.
     *
     * @param s_out / s_grad / s_hess are the three outputs
     * @param s_x / s_x_des / s_Q / s_scratch as above
     */
    template <typename T, bool ACCUMULATE = false, bool MUJOCO_OUTPUT = false>
    __device__
    void quadratic_state_cost_value_grad_hess(T *s_out, T *s_grad, T *s_hess, const T *s_x, const T *s_x_des, const T *s_Q, T *s_scratch) {
        quadratic_state_cost<T>(s_out, s_x, s_x_des, s_Q, s_scratch);
        quadratic_state_cost_gradient<T, ACCUMULATE>(s_grad, s_x, s_x_des, s_Q);
        quadratic_state_cost_hessian<T, ACCUMULATE>(s_hess, s_Q);
    }

    /**
     * quadratic_input_cost: value = 1/2 * sum_i s_R[i] * (s_u[i] - s_u_des[i])^2
     *
     * Notes:
     *   Block-cooperative: each thread accumulates its strided terms into s_scratch, then a serial reduction writes s_out[0].
     *   ACCUMULATE=false overwrites s_out[0]; ACCUMULATE=true ADDS into it (for summing cost terms into one scalar).
     *   s_scratch must hold at least 7 elements.
     *
     * @param s_out is the scalar cost output (s_out[0])
     * @param s_u is the current value (size NUM_VEL = 7)
     * @param s_u_des is the desired/target value (size NUM_VEL = 7)
     * @param s_R is the diagonal weight vector (size NUM_VEL = 7)
     * @param s_scratch is shared scratch of size >= 7
     */
    template <typename T, bool ACCUMULATE = false>
    __device__
    void quadratic_input_cost(T *s_out, const T *s_u, const T *s_u_des, const T *s_R, T *s_scratch) {
        for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < 7; i += blockDim.x*blockDim.y){
            T r = s_u[i] - s_u_des[i];
            s_scratch[i] = static_cast<T>(0.5) * s_R[i] * r * r;
        }
        __syncthreads();
        if(threadIdx.x == 0 && threadIdx.y == 0){
            T acc = static_cast<T>(0);
            for (int i = 0; i < 7; ++i) acc += s_scratch[i];
            if (ACCUMULATE) { s_out[0] += acc; } else { s_out[0] = acc; }
        }
    }

    /**
     * quadratic_input_cost_gradient: g[i] = s_R[i] * (s_u[i] - s_u_des[i])
     *
     * Notes:
     *   ACCUMULATE=false overwrites s_grad; ACCUMULATE=true adds into it (for fusing into a packed [x;u] gradient).
     *
     * @param s_grad is the gradient output (size NUM_VEL = 7)
     * @param s_u / s_u_des / s_R as in the value function
     */
    template <typename T, bool ACCUMULATE = false, bool MUJOCO_OUTPUT = false>
    __device__
    void quadratic_input_cost_gradient(T *s_grad, const T *s_u, const T *s_u_des, const T *s_R) {
        for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < 7; i += blockDim.x*blockDim.y){
            T g = s_R[i] * (s_u[i] - s_u_des[i]);
            if (ACCUMULATE) { s_grad[i] += g; } else { s_grad[i] = g; }
        }
    }

    /**
     * quadratic_input_cost_hessian: Gauss-Newton hessian = diag(s_R) (RATIFIED: GN outer product; for a quadratic cost this is exactly diag(W))
     *
     * Notes:
     *   Writes a dense column-major 7 x 7 matrix; off-diagonal entries are zero.
     *   ACCUMULATE=false overwrites; ACCUMULATE=true adds into the diagonal of an existing block.
     *
     * @param s_hess is the dense hessian output (size 7*7, column-major)
     * @param s_R is the diagonal weight vector (size NUM_VEL = 7)
     */
    template <typename T, bool ACCUMULATE = false, bool MUJOCO_OUTPUT = false>
    __device__
    void quadratic_input_cost_hessian(T *s_hess, const T *s_R) {
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 49; ind += blockDim.x*blockDim.y){
            int row = ind % 7;
            int col = ind / 7;
            T h = (row == col) ? s_R[row] : static_cast<T>(0);
            if (ACCUMULATE) { s_hess[ind] += h; } else { s_hess[ind] = h; }
        }
    }

    /**
     * quadratic_input_cost_value_grad_hess: fused value + gradient + GN-diag hessian in one pass
     *
     * Notes:
     *   Convenience fusion of the three functions above; same conventions and ACCUMULATE semantics for grad/hess.
     *
     * @param s_out / s_grad / s_hess are the three outputs
     * @param s_u / s_u_des / s_R / s_scratch as above
     */
    template <typename T, bool ACCUMULATE = false, bool MUJOCO_OUTPUT = false>
    __device__
    void quadratic_input_cost_value_grad_hess(T *s_out, T *s_grad, T *s_hess, const T *s_u, const T *s_u_des, const T *s_R, T *s_scratch) {
        quadratic_input_cost<T>(s_out, s_u, s_u_des, s_R, s_scratch);
        quadratic_input_cost_gradient<T, ACCUMULATE>(s_grad, s_u, s_u_des, s_R);
        quadratic_input_cost_hessian<T, ACCUMULATE>(s_hess, s_R);
    }

    /**
     * Scalar log-barrier helpers: b = -mu*(log(x-lo)+log(hi-x)); isfinite-guarded so an inf bound contributes zero
     *
     * Notes:
     *   Margin floored at 1e-10 (value) / 1e-6 (grad,hess) to stay finite at the boundary.
     *
     */
    template <typename T> __device__ inline T grid_plant_log_barrier(T x, T lo, T hi, T mu) {
        T b = static_cast<T>(0);
        if (isfinite(lo)) { T d = x - lo; d = (d <= static_cast<T>(1e-10)) ? static_cast<T>(1e-10) : d; b -= log(d); }
        if (isfinite(hi)) { T d = hi - x; d = (d <= static_cast<T>(1e-10)) ? static_cast<T>(1e-10) : d; b -= log(d); }
        return mu * b;
    }
    
    template <typename T> __device__ inline T grid_plant_log_barrier_grad(T x, T lo, T hi, T mu) {
        T g = static_cast<T>(0);
        const T eps = static_cast<T>(1e-6);
        if (isfinite(lo)) { T d = x - lo; T a = (d < static_cast<T>(0)) ? -d : d; if (a < eps) a = eps; d = (d < static_cast<T>(0)) ? -a : a; g -= static_cast<T>(1) / d; }
        if (isfinite(hi)) { T d = hi - x; T a = (d < static_cast<T>(0)) ? -d : d; if (a < eps) a = eps; d = (d < static_cast<T>(0)) ? -a : a; g += static_cast<T>(1) / d; }
        return mu * g;
    }
    
    template <typename T> __device__ inline T grid_plant_log_barrier_hess(T x, T lo, T hi, T mu) {
        T h = static_cast<T>(0);
        const T eps = static_cast<T>(1e-6);
        if (isfinite(lo)) { T d = x - lo; T a = (d < static_cast<T>(0)) ? -d : d; if (a < eps) a = eps; h += static_cast<T>(1) / (a * a); }
        if (isfinite(hi)) { T d = hi - x; T a = (d < static_cast<T>(0)) ? -d : d; if (a < eps) a = eps; h += static_cast<T>(1) / (a * a); }
        return mu * h;
    }
    
    /**
     * joint_position_barrier: value += -mu * sum_i ( log(x_i - lo_i) + log(hi_i - x_i) ) over the NUM_POS q-DOFs
     *
     * Notes:
     *   Block-cooperative reduction into s_scratch then a serial add into s_out[0].
     *   s_lower / s_upper are explicit bound vectors (size 7); an inf entry skips that side (isfinite guard).
     *   s_scratch must hold at least 7 elements.
     *
     * @param s_out is the scalar barrier cost (added into s_out[0])
     * @param s_var is the variable vector (this barrier reads s_var[0 + i])
     * @param s_lower / s_upper are the per-DOF bounds (size 7)
     * @param mu is the barrier weight
     * @param s_scratch is shared scratch of size >= 7
     */
    template <typename T>
    __device__
    void joint_position_barrier(T *s_out, const T *s_var, const T *s_lower, const T *s_upper, const T mu, T *s_scratch) {
        for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < 7; i += blockDim.x*blockDim.y){
            s_scratch[i] = grid_plant_log_barrier<T>(s_var[0 + i], s_lower[i], s_upper[i], mu);
        }
        __syncthreads();
        if(threadIdx.x == 0 && threadIdx.y == 0){
            T acc = static_cast<T>(0);
            for (int i = 0; i < 7; ++i) acc += s_scratch[i];
            s_out[0] += acc;
        }
    }

    /**
     * joint_position_barrier_gradient: s_grad[GRAD_OFFSET + i] += -mu*(1/(x_i-lo_i) - 1/(hi_i-x_i))
     *
     * Notes:
     *   Adds into the packed gradient at GRAD_OFFSET (a template arg so velocity/torque barriers land in the qd / u rows).
     *   isfinite-guarded per side; an unbounded DOF adds exactly zero.
     *
     * @param s_grad is the packed gradient output (added into)
     * @param s_var / s_lower / s_upper / mu as in the value function
     */
    template <typename T, int VAR_OFFSET = 0, int GRAD_OFFSET = 0>
    __device__
    void joint_position_barrier_gradient(T *s_grad, const T *s_var, const T *s_lower, const T *s_upper, const T mu) {
        for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < 7; i += blockDim.x*blockDim.y){
            s_grad[GRAD_OFFSET + i] += grid_plant_log_barrier_grad<T>(s_var[VAR_OFFSET + i], s_lower[i], s_upper[i], mu);
        }
    }

    /**
     * joint_position_barrier_hessian: s_hess[(HESS_OFFSET+i)*HESS_STRIDE + (HESS_OFFSET+i)] += mu*(1/(x_i-lo_i)^2 + 1/(hi_i-x_i)^2)
     *
     * Notes:
     *   Adds the barrier curvature onto the DIAGONAL of a dense column-major HESS_STRIDE x HESS_STRIDE block.
     *   HESS_OFFSET places it in the qd / u block; HESS_STRIDE is the block leading dimension.
     *   isfinite-guarded per side; an unbounded DOF adds exactly zero.
     *
     * @param s_hess is the dense column-major hessian (added into)
     * @param s_var / s_lower / s_upper / mu as above
     */
    template <typename T, int HESS_STRIDE, int VAR_OFFSET = 0, int HESS_OFFSET = 0>
    __device__
    void joint_position_barrier_hessian(T *s_hess, const T *s_var, const T *s_lower, const T *s_upper, const T mu) {
        for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < 7; i += blockDim.x*blockDim.y){
            int d = HESS_OFFSET + i;
            s_hess[d * HESS_STRIDE + d] += grid_plant_log_barrier_hess<T>(s_var[VAR_OFFSET + i], s_lower[i], s_upper[i], mu);
        }
    }

    /**
     * joint_velocity_barrier: value += -mu * sum_i ( log(x_i - lo_i) + log(hi_i - x_i) ) over the NUM_VEL qd-DOFs
     *
     * Notes:
     *   Block-cooperative reduction into s_scratch then a serial add into s_out[0].
     *   s_lower / s_upper are explicit bound vectors (size 7); an inf entry skips that side (isfinite guard).
     *   s_scratch must hold at least 7 elements.
     *
     * @param s_out is the scalar barrier cost (added into s_out[0])
     * @param s_var is the variable vector (this barrier reads s_var[7 + i])
     * @param s_lower / s_upper are the per-DOF bounds (size 7)
     * @param mu is the barrier weight
     * @param s_scratch is shared scratch of size >= 7
     */
    template <typename T>
    __device__
    void joint_velocity_barrier(T *s_out, const T *s_var, const T *s_lower, const T *s_upper, const T mu, T *s_scratch) {
        for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < 7; i += blockDim.x*blockDim.y){
            s_scratch[i] = grid_plant_log_barrier<T>(s_var[7 + i], s_lower[i], s_upper[i], mu);
        }
        __syncthreads();
        if(threadIdx.x == 0 && threadIdx.y == 0){
            T acc = static_cast<T>(0);
            for (int i = 0; i < 7; ++i) acc += s_scratch[i];
            s_out[0] += acc;
        }
    }

    /**
     * joint_velocity_barrier_gradient: s_grad[GRAD_OFFSET + i] += -mu*(1/(x_i-lo_i) - 1/(hi_i-x_i))
     *
     * Notes:
     *   Adds into the packed gradient at GRAD_OFFSET (a template arg so velocity/torque barriers land in the qd / u rows).
     *   isfinite-guarded per side; an unbounded DOF adds exactly zero.
     *
     * @param s_grad is the packed gradient output (added into)
     * @param s_var / s_lower / s_upper / mu as in the value function
     */
    template <typename T, int VAR_OFFSET = 7, int GRAD_OFFSET = 0>
    __device__
    void joint_velocity_barrier_gradient(T *s_grad, const T *s_var, const T *s_lower, const T *s_upper, const T mu) {
        for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < 7; i += blockDim.x*blockDim.y){
            s_grad[GRAD_OFFSET + i] += grid_plant_log_barrier_grad<T>(s_var[VAR_OFFSET + i], s_lower[i], s_upper[i], mu);
        }
    }

    /**
     * joint_velocity_barrier_hessian: s_hess[(HESS_OFFSET+i)*HESS_STRIDE + (HESS_OFFSET+i)] += mu*(1/(x_i-lo_i)^2 + 1/(hi_i-x_i)^2)
     *
     * Notes:
     *   Adds the barrier curvature onto the DIAGONAL of a dense column-major HESS_STRIDE x HESS_STRIDE block.
     *   HESS_OFFSET places it in the qd / u block; HESS_STRIDE is the block leading dimension.
     *   isfinite-guarded per side; an unbounded DOF adds exactly zero.
     *
     * @param s_hess is the dense column-major hessian (added into)
     * @param s_var / s_lower / s_upper / mu as above
     */
    template <typename T, int HESS_STRIDE, int VAR_OFFSET = 7, int HESS_OFFSET = 0>
    __device__
    void joint_velocity_barrier_hessian(T *s_hess, const T *s_var, const T *s_lower, const T *s_upper, const T mu) {
        for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < 7; i += blockDim.x*blockDim.y){
            int d = HESS_OFFSET + i;
            s_hess[d * HESS_STRIDE + d] += grid_plant_log_barrier_hess<T>(s_var[VAR_OFFSET + i], s_lower[i], s_upper[i], mu);
        }
    }

    /**
     * joint_torque_barrier: value += -mu * sum_i ( log(x_i - lo_i) + log(hi_i - x_i) ) over the NUM_VEL u-DOFs
     *
     * Notes:
     *   Block-cooperative reduction into s_scratch then a serial add into s_out[0].
     *   s_lower / s_upper are explicit bound vectors (size 7); an inf entry skips that side (isfinite guard).
     *   s_scratch must hold at least 7 elements.
     *
     * @param s_out is the scalar barrier cost (added into s_out[0])
     * @param s_var is the variable vector (this barrier reads s_var[0 + i])
     * @param s_lower / s_upper are the per-DOF bounds (size 7)
     * @param mu is the barrier weight
     * @param s_scratch is shared scratch of size >= 7
     */
    template <typename T>
    __device__
    void joint_torque_barrier(T *s_out, const T *s_var, const T *s_lower, const T *s_upper, const T mu, T *s_scratch) {
        for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < 7; i += blockDim.x*blockDim.y){
            s_scratch[i] = grid_plant_log_barrier<T>(s_var[0 + i], s_lower[i], s_upper[i], mu);
        }
        __syncthreads();
        if(threadIdx.x == 0 && threadIdx.y == 0){
            T acc = static_cast<T>(0);
            for (int i = 0; i < 7; ++i) acc += s_scratch[i];
            s_out[0] += acc;
        }
    }

    /**
     * joint_torque_barrier_gradient: s_grad[GRAD_OFFSET + i] += -mu*(1/(x_i-lo_i) - 1/(hi_i-x_i))
     *
     * Notes:
     *   Adds into the packed gradient at GRAD_OFFSET (a template arg so velocity/torque barriers land in the qd / u rows).
     *   isfinite-guarded per side; an unbounded DOF adds exactly zero.
     *
     * @param s_grad is the packed gradient output (added into)
     * @param s_var / s_lower / s_upper / mu as in the value function
     */
    template <typename T, int VAR_OFFSET = 0, int GRAD_OFFSET = 0>
    __device__
    void joint_torque_barrier_gradient(T *s_grad, const T *s_var, const T *s_lower, const T *s_upper, const T mu) {
        for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < 7; i += blockDim.x*blockDim.y){
            s_grad[GRAD_OFFSET + i] += grid_plant_log_barrier_grad<T>(s_var[VAR_OFFSET + i], s_lower[i], s_upper[i], mu);
        }
    }

    /**
     * joint_torque_barrier_hessian: s_hess[(HESS_OFFSET+i)*HESS_STRIDE + (HESS_OFFSET+i)] += mu*(1/(x_i-lo_i)^2 + 1/(hi_i-x_i)^2)
     *
     * Notes:
     *   Adds the barrier curvature onto the DIAGONAL of a dense column-major HESS_STRIDE x HESS_STRIDE block.
     *   HESS_OFFSET places it in the qd / u block; HESS_STRIDE is the block leading dimension.
     *   isfinite-guarded per side; an unbounded DOF adds exactly zero.
     *
     * @param s_hess is the dense column-major hessian (added into)
     * @param s_var / s_lower / s_upper / mu as above
     */
    template <typename T, int HESS_STRIDE, int VAR_OFFSET = 0, int HESS_OFFSET = 0>
    __device__
    void joint_torque_barrier_hessian(T *s_hess, const T *s_var, const T *s_lower, const T *s_upper, const T mu) {
        for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < 7; i += blockDim.x*blockDim.y){
            int d = HESS_OFFSET + i;
            s_hess[d * HESS_STRIDE + d] += grid_plant_log_barrier_hess<T>(s_var[VAR_OFFSET + i], s_lower[i], s_upper[i], mu);
        }
    }

    // [grid_plant] plant_step skipped: requires the 'integrator' algorithm (grid::integrator_device) — not generated.
    // [grid_plant] plant_step_gradient[_and_value] skipped: requires 'integrator_gradient' (grid::integrator_gradient_device) — not generated.
    // [grid_plant] plant_step_hessian skipped: requires 'fdsva_so' (grid::integrator_hessian_device) — not generated.
    /**
     * ee_pos: RAW end-effector pose evaluator (no cost coupling; GATO ASK2)
     *
     * Notes:
     *   Caller-scratch INNER: lays out the EE-pose scratch from s_scratch and calls grid::end_effector_pose_inner directly, so it is callable from another kernel's block without aliasing that kernel's dynamic-smem arena.
     *   Fills ALL 1 EE block(s); position is rows 0..2 of each 6-row block.
     *   s_scratch must hold >= END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_COUNT elements of T, 16B aligned.
     *
     * @param s_end_effector_pose is the 6*NUM_EE pose output
     * @param s_q is the joint position vector (size NUM_POS)
     * @param s_scratch is caller shared scratch
     * @param d_robotModel is the GPU model helpers
     */
    template <typename T>
    __device__
    void ee_pos(T *s_end_effector_pose, const T *s_q, T *s_scratch, const grid::robotModel<T> *d_robotModel) {
        using namespace grid;
        // GRID shared arena layout
        //   T s_XmatsHom[144]
        //   T s_temp[32]
        //   bytes s_linalg_smem[GRID_EE_LINALG_SHARED_BYTES<T>()]
        unsigned char *s_arena = reinterpret_cast<unsigned char *>(s_scratch);
        size_t s_arena_offset = 0;
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_XmatsHom = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(144);
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_temp = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(32);
        int *s_topology_helpers = nullptr;
        unsigned char *s_linalg_smem = nullptr;
        if (static_cast<size_t>(GRID_EE_LINALG_SHARED_BYTES<T>()) > 0) {
            s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
            s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
            s_arena_offset += static_cast<size_t>(GRID_EE_LINALG_SHARED_BYTES<T>());
        }
        #ifdef GRID_CUDA_DEBUG_LAYOUT
        assert(s_arena_offset == grid_shared_arena_bytes<T>(176, 0, GRID_EE_LINALG_SHARED_BYTES<T>()));
        #endif
        (void)s_arena_offset;
        load_update_XmatsHom_helpers<T>(s_XmatsHom, s_topology_helpers, s_q, d_robotModel, s_temp);
        end_effector_pose_inner_EE<T, true>(s_end_effector_pose, s_q, s_XmatsHom, s_topology_helpers, s_temp, nullptr, s_linalg_smem);
        __syncthreads();
    }

    /**
     * ee_pos_gradient: RAW end-effector pose + Jacobian evaluator (no cost coupling; GATO ASK2)
     *
     * Notes:
     *   Caller-scratch INNER: ONE XmatsHom load feeds BOTH end_effector_pose_inner and end_effector_pose_gradient_inner (const s_Xhom shared; geometric-Jacobian path, s_dXhom = nullptr) — same single-load structure as ee_pos_cost_gradient.
     *   Jacobian layout: s_end_effector_pose_gradient[6*7*ee + 6*vi + row] (position rows 0..2, orientation rows 3..5; tangent d/dv convention).
     *   s_scratch must hold >= END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_COUNT elements of T, 16B aligned.
     *
     * @param s_end_effector_pose is the 6*NUM_EE pose output
     * @param s_end_effector_pose_gradient is the 6*NUM_VEL*NUM_EE Jacobian output
     * @param s_q is the joint position vector (size NUM_POS)
     * @param s_scratch is caller shared scratch
     * @param d_robotModel is the GPU model helpers
     */
    template <typename T>
    __device__
    void ee_pos_gradient(T *s_end_effector_pose, T *s_end_effector_pose_gradient, const T *s_q, T *s_scratch, const grid::robotModel<T> *d_robotModel) {
        using namespace grid;
        // GRID shared arena layout
        //   T s_XmatsHom[144]
        //   T s_temp[190]
        //   bytes s_linalg_smem[GRID_EE_LINALG_SHARED_BYTES<T>()]
        unsigned char *s_arena = reinterpret_cast<unsigned char *>(s_scratch);
        size_t s_arena_offset = 0;
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_XmatsHom = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(144);
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_temp = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(190);
        int *s_topology_helpers = nullptr;
        unsigned char *s_linalg_smem = nullptr;
        if (static_cast<size_t>(GRID_EE_LINALG_SHARED_BYTES<T>()) > 0) {
            s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
            s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
            s_arena_offset += static_cast<size_t>(GRID_EE_LINALG_SHARED_BYTES<T>());
        }
        #ifdef GRID_CUDA_DEBUG_LAYOUT
        assert(s_arena_offset == grid_shared_arena_bytes<T>(334, 0, GRID_EE_LINALG_SHARED_BYTES<T>()));
        #endif
        (void)s_arena_offset;
        load_update_XmatsHom_helpers<T>(s_XmatsHom, s_topology_helpers, s_q, d_robotModel, s_temp);
        end_effector_pose_inner_EE<T, true>(s_end_effector_pose, s_q, s_XmatsHom, s_topology_helpers, s_temp, nullptr, s_linalg_smem);
        __syncthreads();
        end_effector_pose_gradient_inner_EE<T, true>(s_end_effector_pose_gradient, s_q, s_XmatsHom, nullptr, s_topology_helpers, s_temp, nullptr, s_linalg_smem);
        __syncthreads();
    }

    /**
     * ee_pos_cost: value = 1/2 * sum_r W[r] * (p_r(q) - p_des_r)^2 over the 3 position axes
     *
     * Notes:
     *   Caller-scratch INNER: lays out the EE-pose scratch from s_scratch and calls grid::end_effector_pose_inner directly (NOT the auto-allocating _device), so it is callable from another kernel's block without aliasing that kernel's dynamic-smem arena.
     *   EE selects which end-effector (0..0).
     *   s_end_effector_pose must hold 6*NUM_EE; s_scratch must hold >= END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_COUNT elements of T.
     *
     * @param s_out is the scalar cost output (s_out[0])
     * @param s_q is the joint position vector (size NUM_POS)
     * @param s_p_des is the desired EE position (3-vector)
     * @param s_W is the per-axis position weight (3-vector)
     * @param s_end_effector_pose is scratch for the 6*NUM_EE pose (the position is rows 0..2 of EE block)
     * @param s_scratch is caller shared scratch for the EE-pose helper (>= END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_COUNT, 16B aligned)
     * @param d_robotModel is the GPU model helpers
     */
    template <typename T, int EE = 0, bool ACCUMULATE = false>
    __device__
    void ee_pos_cost(T *s_out, const T *s_q, const T *s_p_des, const T *s_W, T *s_end_effector_pose, T *s_scratch, const grid::robotModel<T> *d_robotModel) {
        using namespace grid;
        // GRID shared arena layout
        //   T s_XmatsHom[144]
        //   T s_temp[32]
        //   bytes s_linalg_smem[GRID_EE_LINALG_SHARED_BYTES<T>()]
        unsigned char *s_arena = reinterpret_cast<unsigned char *>(s_scratch);
        size_t s_arena_offset = 0;
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_XmatsHom = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(144);
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_temp = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(32);
        int *s_topology_helpers = nullptr;
        unsigned char *s_linalg_smem = nullptr;
        if (static_cast<size_t>(GRID_EE_LINALG_SHARED_BYTES<T>()) > 0) {
            s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
            s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
            s_arena_offset += static_cast<size_t>(GRID_EE_LINALG_SHARED_BYTES<T>());
        }
        #ifdef GRID_CUDA_DEBUG_LAYOUT
        assert(s_arena_offset == grid_shared_arena_bytes<T>(176, 0, GRID_EE_LINALG_SHARED_BYTES<T>()));
        #endif
        (void)s_arena_offset;
        load_update_XmatsHom_helpers<T>(s_XmatsHom, s_topology_helpers, s_q, d_robotModel, s_temp);
        end_effector_pose_inner_EE<T, true>(s_end_effector_pose, s_q, s_XmatsHom, s_topology_helpers, s_temp, nullptr, s_linalg_smem);
        __syncthreads();
        if(threadIdx.x == 0 && threadIdx.y == 0){
            T acc = static_cast<T>(0);
            #pragma unroll
            for (int r = 0; r < 3; ++r) { T e = s_end_effector_pose[6*EE + r] - s_p_des[r]; acc += static_cast<T>(0.5) * s_W[r] * e * e; }
            if (ACCUMULATE) { s_out[0] += acc; } else { s_out[0] = acc; }
        }
    }

    /**
     * ee_pos_cost_gradient: grad_x = [J_p^T W (p - p_des) ; 0], over x = [q; qd]
     *
     * Notes:
     *   Caller-scratch INNER: ONE XmatsHom load feeds BOTH end_effector_pose_inner (for p) and end_effector_pose_gradient_inner (for J_p) -- s_Xhom is const in both inners, so the local homogeneous transforms are loaded once and shared (vs the old double-load through two auto-allocating _device calls). The geometric-Jacobian inner uses only s_Xhom (s_dXhom = nullptr), so no per-joint d-transform load is needed. Callable from another kernel's block without aliasing that kernel's dynamic-smem arena.
     *   J_p = rows 0..2 of s_end_effector_pose_gradient, layout s_end_effector_pose_gradient[6*NUM_VEL*ee + 6*vi + row].
     *   The qd-block of the gradient (entries NUM_VEL..13) is set to exactly zero.
     *   ACCUMULATE=false overwrites s_grad; true adds (for fusing with a state-cost gradient).
     *
     * @param s_grad is the gradient over x (size NUM_POS + NUM_VEL = 14)
     * @param s_q / s_p_des / s_W / d_robotModel as above
     * @param s_end_effector_pose is 6*NUM_EE pose scratch; s_end_effector_pose_gradient is 6*NUM_VEL*NUM_EE Jacobian scratch
     * @param s_scratch is caller shared scratch for the EE-pose+gradient helper (>= END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_COUNT, 16B aligned)
     */
    template <typename T, int EE = 0, bool ACCUMULATE = false, bool MUJOCO_OUTPUT = false>
    __device__
    void ee_pos_cost_gradient(T *s_grad, const T *s_q, const T *s_p_des, const T *s_W, T *s_end_effector_pose, T *s_end_effector_pose_gradient, T *s_scratch, const grid::robotModel<T> *d_robotModel) {
        using namespace grid;
        // GRID shared arena layout
        //   T s_XmatsHom[144]
        //   T s_temp[190]
        //   bytes s_linalg_smem[GRID_EE_LINALG_SHARED_BYTES<T>()]
        unsigned char *s_arena = reinterpret_cast<unsigned char *>(s_scratch);
        size_t s_arena_offset = 0;
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_XmatsHom = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(144);
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_temp = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(190);
        int *s_topology_helpers = nullptr;
        unsigned char *s_linalg_smem = nullptr;
        if (static_cast<size_t>(GRID_EE_LINALG_SHARED_BYTES<T>()) > 0) {
            s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
            s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
            s_arena_offset += static_cast<size_t>(GRID_EE_LINALG_SHARED_BYTES<T>());
        }
        #ifdef GRID_CUDA_DEBUG_LAYOUT
        assert(s_arena_offset == grid_shared_arena_bytes<T>(334, 0, GRID_EE_LINALG_SHARED_BYTES<T>()));
        #endif
        (void)s_arena_offset;
        load_update_XmatsHom_helpers<T>(s_XmatsHom, s_topology_helpers, s_q, d_robotModel, s_temp);
        end_effector_pose_inner_EE<T, true>(s_end_effector_pose, s_q, s_XmatsHom, s_topology_helpers, s_temp, nullptr, s_linalg_smem);
        __syncthreads();
        end_effector_pose_gradient_inner_EE<T, true>(s_end_effector_pose_gradient, s_q, s_XmatsHom, nullptr, s_topology_helpers, s_temp, nullptr, s_linalg_smem);
        __syncthreads();
        for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < 7; i += blockDim.x*blockDim.y){
            T g = static_cast<T>(0);
            #pragma unroll
            for (int r = 0; r < 3; ++r) {
                T Jri = s_end_effector_pose_gradient[6*7*EE + 6*i + r];
                T e   = s_end_effector_pose[6*EE + r] - s_p_des[r];
                g += Jri * s_W[r] * e;
            }
            if (ACCUMULATE) { s_grad[i] += g; } else { s_grad[i] = g; }
        }
        if (!ACCUMULATE) {
            for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < 7; i += blockDim.x*blockDim.y){
                s_grad[7 + i] = static_cast<T>(0);
            }
        }
    }

    /**
     * ee_pos_cost_hessian: EE-position cost hessian. DEFAULT = full Newton = J_p^T W J_p + sum_r W[r] (p_r - p_des_r) d2p_r/dv2; GAUSS_NEWTON=true = ratified PSD GN term only
     *
     * Notes:
     *   DEFAULT (GAUSS_NEWTON=false) folds the exact residual-weighted EE-curvature term via the analytic d2ee (grid::end_effector_pose_hessian_inner). Not necessarily PSD away from the solution -- callers must regularize (e.g. the solver's rho schedule).
     *   GAUSS_NEWTON=true keeps the ratified PSD choice H = J_p^T diag(W) J_p (curvature dropped); s_p_des, s_end_effector_pose and s_end_effector_pose_hessian may be nullptr in that case.
     *   PSD_CLAMP=true (opt-in, default false) eigen-clamps the NV x NV q-block to >= psd_reg_eps (glass::eig_clamp) so the returned hessian is SPD and directly factorable even when the Newton curvature is indefinite -- a guaranteed-PSD alternative to a caller-side rho schedule. Costs one block-cooperative Jacobi eigensolve; s_scratch must hold NV*NV + eig_clamp_scratch when set.
     *   Caller-scratch INNER: ONE XmatsHom load feeds the needed inners (GN: gradient_inner only, s_dXhom = nullptr; Newton: pose_inner for the residual + hessian_inner, which also fills the gradient buffer), so it is callable from another kernel's block without aliasing that kernel's dynamic-smem arena.
     *   d2p layout: s_end_effector_pose_hessian[6*NV*NV*ee + r*NV*NV + vi*NV + vj] (pose row r, joint pair (vi, vj); tangent d/dv convention, position rows 0..2 symmetric in (vi, vj)).
     *   Dense column-major NX x NX (NX = NUM_POS + NUM_VEL = 14); only the top-left NUM_VEL x NUM_VEL q-block is non-zero.
     *   ACCUMULATE=false overwrites the whole NX x NX block; true adds the q-block into an existing hessian.
     *
     * @param s_hess is the dense x-hessian output (size 14*14, column-major)
     * @param s_q / s_p_des / s_W / d_robotModel as above (s_p_des is unused under GAUSS_NEWTON)
     * @param s_end_effector_pose is 6*NUM_EE pose scratch (Newton only); s_end_effector_pose_gradient is 6*NUM_VEL*NUM_EE Jacobian scratch; s_end_effector_pose_hessian is 6*NUM_VEL*NUM_VEL*NUM_EE d2ee scratch (Newton only)
     * @param s_scratch is caller shared scratch for the EE helpers (GN: >= END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_COUNT; Newton: >= END_EFFECTOR_POSE_HESSIAN_DYNAMIC_SHARED_MEM_COUNT covers it; 16B aligned)
     */
    template <typename T, int EE = 0, bool ACCUMULATE = false, bool GAUSS_NEWTON = false, bool MUJOCO_OUTPUT = false, bool PSD_CLAMP = false>
    __device__
    void ee_pos_cost_hessian(T *s_hess, const T *s_q, const T *s_p_des, const T *s_W, T *s_end_effector_pose, T *s_end_effector_pose_gradient, T *s_end_effector_pose_hessian, T *s_scratch, const grid::robotModel<T> *d_robotModel, T psd_reg_eps = static_cast<T>(1e-6)) {
        using namespace grid;
        static_assert(GAUSS_NEWTON, "full-Newton ee_pos_cost_hessian requires 'end_effector_pose_hessian' in the algorithm set; pass GAUSS_NEWTON=true for the J_p^T W J_p approximation");
        // GRID shared arena layout
        //   T s_XmatsHom[144]
        //   T s_temp[190]
        //   bytes s_linalg_smem[GRID_EE_LINALG_SHARED_BYTES<T>()]
        unsigned char *s_arena = reinterpret_cast<unsigned char *>(s_scratch);
        size_t s_arena_offset = 0;
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_XmatsHom = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(144);
        s_arena_offset = grid_align_up(s_arena_offset, alignof(T));
        T *s_temp = grid_arena_ptr<T>(s_arena, s_arena_offset);
        s_arena_offset += sizeof(T) * static_cast<size_t>(190);
        int *s_topology_helpers = nullptr;
        unsigned char *s_linalg_smem = nullptr;
        if (static_cast<size_t>(GRID_EE_LINALG_SHARED_BYTES<T>()) > 0) {
            s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));
            s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);
            s_arena_offset += static_cast<size_t>(GRID_EE_LINALG_SHARED_BYTES<T>());
        }
        #ifdef GRID_CUDA_DEBUG_LAYOUT
        assert(s_arena_offset == grid_shared_arena_bytes<T>(334, 0, GRID_EE_LINALG_SHARED_BYTES<T>()));
        #endif
        (void)s_arena_offset;
        load_update_XmatsHom_helpers<T>(s_XmatsHom, s_topology_helpers, s_q, d_robotModel, s_temp);
        end_effector_pose_gradient_inner_EE<T, true>(s_end_effector_pose_gradient, s_q, s_XmatsHom, nullptr, s_topology_helpers, s_temp, nullptr, s_linalg_smem);
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 196; ind += blockDim.x*blockDim.y){
            int row = ind % 14;
            int col = ind / 14;
            T h = static_cast<T>(0);
            if (row < 7 && col < 7) {
                #pragma unroll
                for (int r = 0; r < 3; ++r) {
                    T Jri = s_end_effector_pose_gradient[6*7*EE + 6*row + r];
                    T Jrj = s_end_effector_pose_gradient[6*7*EE + 6*col + r];
                    h += Jri * s_W[r] * Jrj;
                }
            }
            if (ACCUMULATE) { s_hess[ind] += h; } else { s_hess[ind] = h; }
        }
        if constexpr (PSD_CLAMP) {
            __syncthreads();
            T *s_psd_qb = s_scratch; T *s_psd_eig = &s_scratch[49];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 49; ind += blockDim.x*blockDim.y){
                int r = ind % 7; int c = ind / 7;
                s_psd_qb[r + 7*c] = s_hess[r + 14*c];
            }
            __syncthreads();
            glass::eig_clamp<T, 7>(s_psd_qb, psd_reg_eps, s_psd_eig);
            __syncthreads();
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 49; ind += blockDim.x*blockDim.y){
                int r = ind % 7; int c = ind / 7;
                s_hess[r + 14*c] = s_psd_qb[r + 7*c];
            }
            __syncthreads();
        }
    }

    /**
     * tracking_cost: total scalar cost = ee_pos_cost + quadratic_state_cost + quadratic_input_cost + joint_{position,velocity,torque}_barrier (GATO BSQP recipe, composed from the per-term inners)
     *
     * Notes:
     *   PRESET (sugar): one composition of the public per-term cost inners; users compose other mixes directly.
     *   ACCUMULATE=false overwrites s_out[0]; true adds (the first term carries ACCUMULATE, the rest add).
     *   s_end_effector_pose holds 6*NUM_EE; s_scratch must hold >= END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_COUNT (and >= NX).
     *   Fixed-base only (NUM_POS == NUM_VEL); floating-base composition is a follow-up.
     *
     * @param s_out is the scalar total-cost output (s_out[0])
     * @param s_x / s_u are the current state [q; qd] and control (sizes 14 / 7)
     * @param s_x_des / s_u_des / s_ee_des are the targets (state 14, input 7, EE position 3)
     * @param s_Q / s_R are the quadratic state/input diagonal weights (sizes 14 / 7); s_W is the per-axis EE position weight (3). Zero a weight to disable that term.
     * @param s_q_lower/upper + mu_q, s_qd_lower/upper + mu_qd, s_u_lower/upper + mu_u are the position / velocity / torque log-barrier bounds + weights (mu=0 or +/-inf bound disables).
     * @param EE selects the end-effector; running-vs-terminal weighting is caller-supplied (write terminal weights at the terminal knot — contract, not a baked branch).
     * @param s_end_effector_pose / s_scratch are caller EE-pose scratch; d_robotModel is the GPU model
     */
    template <typename T, int EE = 0, bool ACCUMULATE = false>
    __device__
    void tracking_cost(T *s_out, const T *s_x, const T *s_u, const T *s_x_des, const T *s_u_des, const T *s_ee_des, const T *s_Q, const T *s_R, const T *s_W, const T *s_q_lower, const T *s_q_upper, const T mu_q, const T *s_qd_lower, const T *s_qd_upper, const T mu_qd, const T *s_u_lower, const T *s_u_upper, const T mu_u, T *s_end_effector_pose, T *s_scratch, const grid::robotModel<T> *d_robotModel) {
        ee_pos_cost<T, EE, ACCUMULATE>(s_out, s_x, s_ee_des, s_W, s_end_effector_pose, s_scratch, d_robotModel);
        __syncthreads();
        quadratic_state_cost<T, true>(s_out, s_x, s_x_des, s_Q, s_scratch);
        __syncthreads();
        quadratic_input_cost<T, true>(s_out, s_u, s_u_des, s_R, s_scratch);
        __syncthreads();
        joint_position_barrier<T>(s_out, s_x, s_q_lower, s_q_upper, mu_q, s_scratch);
        __syncthreads();
        joint_velocity_barrier<T>(s_out, s_x, s_qd_lower, s_qd_upper, mu_qd, s_scratch);
        __syncthreads();
        joint_torque_barrier<T>(s_out, s_u, s_u_lower, s_u_upper, mu_u, s_scratch);
    }

    /**
     * tracking_cost_gradient: s_qk (state gradient, NX) + s_rk (input gradient, NU), composed from the per-term gradient inners
     *
     * Notes:
     *   State block s_qk = ee_pos_cost_gradient (J^T W r in q-block, 0 in qd) + quadratic_state_cost_gradient + joint_position_barrier_gradient (q-block) + joint_velocity_barrier_gradient (qd-block).
     *   Input block s_rk = quadratic_input_cost_gradient + joint_torque_barrier_gradient.
     *   ACCUMULATE=false overwrites the blocks; true adds into them (the EE / input-quadratic terms carry ACCUMULATE, the rest add).
     *   Sync between every accumulating term (different inners own different threads per index).
     *   s_scratch must hold >= END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_COUNT. Fixed-base only.
     *
     * @param s_qk is the state-block gradient output (size NX = 14)
     * @param s_rk is the input-block gradient output (size NU = 7)
     * @param s_x / s_u are the current state [q; qd] and control (sizes 14 / 7)
     * @param s_x_des / s_u_des / s_ee_des are the targets (state 14, input 7, EE position 3)
     * @param s_Q / s_R are the quadratic state/input diagonal weights (sizes 14 / 7); s_W is the per-axis EE position weight (3). Zero a weight to disable that term.
     * @param s_q_lower/upper + mu_q, s_qd_lower/upper + mu_qd, s_u_lower/upper + mu_u are the position / velocity / torque log-barrier bounds + weights (mu=0 or +/-inf bound disables).
     * @param EE selects the end-effector; running-vs-terminal weighting is caller-supplied (write terminal weights at the terminal knot — contract, not a baked branch).
     * @param s_end_effector_pose (6*NUM_EE) / s_end_effector_pose_gradient (6*NUM_VEL*NUM_EE) / s_scratch are caller EE scratch
     * @param d_robotModel is the GPU model
     */
    template <typename T, int EE = 0, bool ACCUMULATE = false>
    __device__
    void tracking_cost_gradient(T *s_qk, T *s_rk, const T *s_x, const T *s_u, const T *s_x_des, const T *s_u_des, const T *s_ee_des, const T *s_Q, const T *s_R, const T *s_W, const T *s_q_lower, const T *s_q_upper, const T mu_q, const T *s_qd_lower, const T *s_qd_upper, const T mu_qd, const T *s_u_lower, const T *s_u_upper, const T mu_u, T *s_end_effector_pose, T *s_end_effector_pose_gradient, T *s_scratch, const grid::robotModel<T> *d_robotModel) {
        // ---- state-block gradient s_qk (NX) ----
        ee_pos_cost_gradient<T, EE, ACCUMULATE>(s_qk, s_x, s_ee_des, s_W, s_end_effector_pose, s_end_effector_pose_gradient, s_scratch, d_robotModel);
        __syncthreads();
        quadratic_state_cost_gradient<T, true>(s_qk, s_x, s_x_des, s_Q);
        __syncthreads();
        joint_position_barrier_gradient<T, 0, 0>(s_qk, s_x, s_q_lower, s_q_upper, mu_q);
        __syncthreads();
        joint_velocity_barrier_gradient<T, 7, 7>(s_qk, s_x, s_qd_lower, s_qd_upper, mu_qd);
        __syncthreads();
        // ---- input-block gradient s_rk (NU) ----
        quadratic_input_cost_gradient<T, ACCUMULATE>(s_rk, s_u, s_u_des, s_R);
        __syncthreads();
        joint_torque_barrier_gradient<T, 0, 0>(s_rk, s_u, s_u_lower, s_u_upper, mu_u);
    }

    /**
     * tracking_cost_hessian: s_Qk (state GN hessian, NX*NX col-major) + s_Rk (input hessian, NU*NU), composed from the per-term hessian inners
     *
     * Notes:
     *   State block s_Qk = ee_pos_cost_hessian (J^T W J in q-block) + quadratic_state_cost_hessian (diag) + joint_position_barrier_hessian (q diagonal) + joint_velocity_barrier_hessian (qd diagonal).
     *   Input block s_Rk = quadratic_input_cost_hessian (diag) + joint_torque_barrier_hessian (diagonal).
     *   Gauss-Newton: the EE value-curvature is dropped (matches the per-term GN choice).
     *   ACCUMULATE=false overwrites; true adds (the EE / input-quadratic terms carry ACCUMULATE, the rest add).
     *   s_scratch must hold >= END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_COUNT. Fixed-base only.
     *
     * @param s_Qk is the state GN hessian output (size NX*NX = 196, column-major)
     * @param s_Rk is the input hessian output (size NU*NU = 49, column-major)
     * @param s_x / s_u are the current state and control
     * @param s_Q / s_R / s_W are the quadratic state/input + EE weights
     * @param s_q_lower/upper + mu_q, s_qd_lower/upper + mu_qd, s_u_lower/upper + mu_u are the barrier bounds + weights
     * @param s_end_effector_pose_gradient (6*NUM_VEL*NUM_EE) / s_scratch are caller EE scratch; d_robotModel is the GPU model
     */
    template <typename T, int EE = 0, bool ACCUMULATE = false>
    __device__
    void tracking_cost_hessian(T *s_Qk, T *s_Rk, const T *s_x, const T *s_u, const T *s_Q, const T *s_R, const T *s_W, const T *s_q_lower, const T *s_q_upper, const T mu_q, const T *s_qd_lower, const T *s_qd_upper, const T mu_qd, const T *s_u_lower, const T *s_u_upper, const T mu_u, T *s_end_effector_pose_gradient, T *s_scratch, const grid::robotModel<T> *d_robotModel) {
        // ---- state-block GN hessian s_Qk (NX*NX) ----
        ee_pos_cost_hessian<T, EE, ACCUMULATE, true>(s_Qk, s_x, nullptr, s_W, nullptr, s_end_effector_pose_gradient, nullptr, s_scratch, d_robotModel);
        __syncthreads();
        quadratic_state_cost_hessian<T, true>(s_Qk, s_Q);
        __syncthreads();
        joint_position_barrier_hessian<T, 14, 0, 0>(s_Qk, s_x, s_q_lower, s_q_upper, mu_q);
        __syncthreads();
        joint_velocity_barrier_hessian<T, 14, 7, 7>(s_Qk, s_x, s_qd_lower, s_qd_upper, mu_qd);
        __syncthreads();
        // ---- input-block hessian s_Rk (NU*NU) ----
        quadratic_input_cost_hessian<T, ACCUMULATE>(s_Rk, s_R);
        __syncthreads();
        joint_torque_barrier_hessian<T, 7, 0, 0>(s_Rk, s_u, s_u_lower, s_u_upper, mu_u);
    }

    // [grid_plant] com_cost skipped: requires 'com'+'ccrba'.
    // [grid_plant] momentum_cost skipped: requires 'dccrba' for the full tangent-state Jacobian.
    /**
     * quadratic_state_cost_kernel: value + gradient + GN-diag hessian per timestep
     *
     * @param d_out scalar cost (1 per timestep)
     * @param d_grad gradient (14 per timestep)
     * @param d_hess dense col-major hessian (14*14 per timestep)
     * @param d_x / d_x_des / d_Q inputs (14 per timestep)
     * @param NUM_TIMESTEPS is the batch size
     */
    template <typename T, bool MUJOCO_OUTPUT = false>
    __global__
    void quadratic_state_cost_kernel(T *d_out, T *d_grad, T *d_hess, const T *d_x, const T *d_x_des, const T *d_Q, const int NUM_TIMESTEPS) {
        __shared__ T s_scratch[14];
        for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
            quadratic_state_cost_value_grad_hess<T>(&d_out[k], &d_grad[k*14], &d_hess[k*196], &d_x[k*14], &d_x_des[k*14], &d_Q[k*14], s_scratch);
            __syncthreads();
        }
    }

    /**
     * quadratic_input_cost_kernel: value + gradient + GN-diag hessian per timestep
     *
     * @param d_out scalar cost (1 per timestep)
     * @param d_grad gradient (7 per timestep)
     * @param d_hess dense col-major hessian (7*7 per timestep)
     * @param d_u / d_u_des / d_R inputs (7 per timestep)
     * @param NUM_TIMESTEPS is the batch size
     */
    template <typename T, bool MUJOCO_OUTPUT = false>
    __global__
    void quadratic_input_cost_kernel(T *d_out, T *d_grad, T *d_hess, const T *d_u, const T *d_u_des, const T *d_R, const int NUM_TIMESTEPS) {
        __shared__ T s_scratch[7];
        for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
            quadratic_input_cost_value_grad_hess<T>(&d_out[k], &d_grad[k*7], &d_hess[k*49], &d_u[k*7], &d_u_des[k*7], &d_R[k*7], s_scratch);
            __syncthreads();
        }
    }

    /**
     * joint_position_barrier_kernel: value + grad + hess-diagonal per timestep
     *
     * @param d_out scalar barrier cost (1 per timestep)
     * @param d_grad per-DOF gradient (7 per timestep)
     * @param d_hess_diag per-DOF hessian diagonal (7 per timestep)
     * @param d_var variable vector (7 per timestep)
     * @param d_lower / d_upper per-DOF bounds (7 per timestep)
     * @param mu barrier weight; NUM_TIMESTEPS is the batch size
     */
    template <typename T>
    __global__
    void joint_position_barrier_kernel(T *d_out, T *d_grad, T *d_hess_diag, const T *d_var, const T *d_lower, const T *d_upper, const T mu, const int NUM_TIMESTEPS) {
        __shared__ T s_scratch[7];
        for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
            if(threadIdx.x == 0 && threadIdx.y == 0){
                d_out[k] = static_cast<T>(0);
            }
            __syncthreads();
            const T *s_var = &d_var[k*7]; const T *s_lo = &d_lower[k*7]; const T *s_hi = &d_upper[k*7];
            joint_position_barrier<T>(&d_out[k], s_var, s_lo, s_hi, mu, s_scratch);
            __syncthreads();
            for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < 7; i += blockDim.x*blockDim.y){
                d_grad[k*7 + i] = grid_plant_log_barrier_grad<T>(s_var[i], s_lo[i], s_hi[i], mu);
                d_hess_diag[k*7 + i] = grid_plant_log_barrier_hess<T>(s_var[i], s_lo[i], s_hi[i], mu);
            }
            __syncthreads();
        }
    }

    /**
     * joint_velocity_barrier_kernel: value + grad + hess-diagonal per timestep
     *
     * @param d_out scalar barrier cost (1 per timestep)
     * @param d_grad per-DOF gradient (7 per timestep)
     * @param d_hess_diag per-DOF hessian diagonal (7 per timestep)
     * @param d_var variable vector (7 per timestep)
     * @param d_lower / d_upper per-DOF bounds (7 per timestep)
     * @param mu barrier weight; NUM_TIMESTEPS is the batch size
     */
    template <typename T>
    __global__
    void joint_velocity_barrier_kernel(T *d_out, T *d_grad, T *d_hess_diag, const T *d_var, const T *d_lower, const T *d_upper, const T mu, const int NUM_TIMESTEPS) {
        __shared__ T s_scratch[7];
        for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
            if(threadIdx.x == 0 && threadIdx.y == 0){
                d_out[k] = static_cast<T>(0);
            }
            __syncthreads();
            const T *s_var = &d_var[k*7]; const T *s_lo = &d_lower[k*7]; const T *s_hi = &d_upper[k*7];
            joint_velocity_barrier<T>(&d_out[k], s_var, s_lo, s_hi, mu, s_scratch);
            __syncthreads();
            for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < 7; i += blockDim.x*blockDim.y){
                d_grad[k*7 + i] = grid_plant_log_barrier_grad<T>(s_var[i], s_lo[i], s_hi[i], mu);
                d_hess_diag[k*7 + i] = grid_plant_log_barrier_hess<T>(s_var[i], s_lo[i], s_hi[i], mu);
            }
            __syncthreads();
        }
    }

    /**
     * joint_torque_barrier_kernel: value + grad + hess-diagonal per timestep
     *
     * @param d_out scalar barrier cost (1 per timestep)
     * @param d_grad per-DOF gradient (7 per timestep)
     * @param d_hess_diag per-DOF hessian diagonal (7 per timestep)
     * @param d_var variable vector (7 per timestep)
     * @param d_lower / d_upper per-DOF bounds (7 per timestep)
     * @param mu barrier weight; NUM_TIMESTEPS is the batch size
     */
    template <typename T>
    __global__
    void joint_torque_barrier_kernel(T *d_out, T *d_grad, T *d_hess_diag, const T *d_var, const T *d_lower, const T *d_upper, const T mu, const int NUM_TIMESTEPS) {
        __shared__ T s_scratch[7];
        for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
            if(threadIdx.x == 0 && threadIdx.y == 0){
                d_out[k] = static_cast<T>(0);
            }
            __syncthreads();
            const T *s_var = &d_var[k*7]; const T *s_lo = &d_lower[k*7]; const T *s_hi = &d_upper[k*7];
            joint_torque_barrier<T>(&d_out[k], s_var, s_lo, s_hi, mu, s_scratch);
            __syncthreads();
            for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < 7; i += blockDim.x*blockDim.y){
                d_grad[k*7 + i] = grid_plant_log_barrier_grad<T>(s_var[i], s_lo[i], s_hi[i], mu);
                d_hess_diag[k*7 + i] = grid_plant_log_barrier_hess<T>(s_var[i], s_lo[i], s_hi[i], mu);
            }
            __syncthreads();
        }
    }

    /**
     * ee_pos_cost_kernel: value + grad_x + GN hess_x per timestep (EE=0)
     *
     * @param d_out scalar cost (1 per timestep)
     * @param d_grad grad over x (14 per timestep)
     * @param d_hess dense col-major x-hessian (196 per timestep)
     * @param d_q joint positions (NUM_POS per timestep)
     * @param d_p_des desired EE position (3 per timestep)
     * @param d_W per-axis weight (3 per timestep)
     * @param d_end_effector_pose / d_end_effector_pose_gradient global scratch (6*NUM_EES / 6*NUM_VEL*NUM_EES per timestep)
     * @param NUM_TIMESTEPS is the batch size
     */
    template <typename T, int EE = 0, bool MUJOCO_OUTPUT = false>
    __global__
    void ee_pos_cost_kernel(T *d_out, T *d_grad, T *d_hess, const T *d_q, const T *d_p_des, const T *d_W, T *d_end_effector_pose, T *d_end_effector_pose_gradient, const grid::robotModel<T> *d_robotModel, const int NUM_TIMESTEPS) {
        extern __shared__ __align__(16) T s_ee_arena[];
        for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
            const T *s_q = &d_q[k*7]; const T *s_p_des = &d_p_des[k*3]; const T *s_W = &d_W[k*3];
            T *s_end_effector_pose = &d_end_effector_pose[k*6]; T *s_end_effector_pose_gradient = &d_end_effector_pose_gradient[k*42];
            ee_pos_cost<T, EE>(&d_out[k], s_q, s_p_des, s_W, s_end_effector_pose, s_ee_arena, d_robotModel);
            __syncthreads();
            ee_pos_cost_gradient<T, EE>(&d_grad[k*14], s_q, s_p_des, s_W, s_end_effector_pose, s_end_effector_pose_gradient, s_ee_arena, d_robotModel);
            __syncthreads();
            ee_pos_cost_hessian<T, EE, false, true>(&d_hess[k*196], s_q, s_p_des, s_W, s_end_effector_pose, s_end_effector_pose_gradient, nullptr, s_ee_arena, d_robotModel);
            __syncthreads();
        }
    }

    #define GRID_PLANT_HAS_EE_COST 1
    #define GRID_PLANT_HAS_TRACKING_COST 1
}
