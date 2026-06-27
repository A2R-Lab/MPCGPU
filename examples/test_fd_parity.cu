// Dynamics parity: gato_plant::forwardDynamics (adapter) vs grid::forward_dynamics_device
// (grid's own full wrapper) on random states. |qdd_adapter - qdd_ref| must be ~0.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cooperative_groups.h>
#include "dynamics/rbd_plant.cuh"

template<typename T>
__global__ void k_adapter(T* d_qdd, const T* d_q, const T* d_qd, const T* d_u, void* d_rm){
    extern __shared__ T s[];
    __shared__ T s_q[7], s_qd[7], s_u[7], s_qdd[7];
    for(int i=threadIdx.x;i<7;i+=blockDim.x){ s_q[i]=d_q[i]; s_qd[i]=d_qd[i]; s_u[i]=d_u[i]; }
    __syncthreads();
    gato_plant::forwardDynamics<T>(s_qdd, s_q, s_qd, s_u, s, d_rm, cooperative_groups::this_thread_block());
    __syncthreads();
    for(int i=threadIdx.x;i<7;i+=blockDim.x) d_qdd[i]=s_qdd[i];
}

template<typename T>
__global__ void k_ref(T* d_qdd, const T* d_q, const T* d_qd, const T* d_u, void* d_rm){
    __shared__ T s_q[7], s_qd[7], s_u[7], s_qdd[7];
    for(int i=threadIdx.x;i<7;i+=blockDim.x){ s_q[i]=d_q[i]; s_qd[i]=d_qd[i]; s_u[i]=d_u[i]; }
    __syncthreads();
    grid::forward_dynamics_device<T>(s_qdd, s_q, s_qd, s_u, (grid::robotModel<T>*)d_rm, /*d_f_ext*/nullptr, gato_plant::GRAVITY<T>());
    __syncthreads();
    for(int i=threadIdx.x;i<7;i+=blockDim.x) d_qdd[i]=s_qdd[i];
}

int main(){
    using T = float;
    srand(7);
    auto *d_rm = grid::init_robotModel<T>();
    T *d_q,*d_qd,*d_u,*d_qa,*d_qr;
    cudaMalloc(&d_q,7*sizeof(T)); cudaMalloc(&d_qd,7*sizeof(T)); cudaMalloc(&d_u,7*sizeof(T));
    cudaMalloc(&d_qa,7*sizeof(T)); cudaMalloc(&d_qr,7*sizeof(T));
    size_t smem = grid::FORWARD_DYNAMICS_DYNAMIC_SHARED_MEM_COUNT*sizeof(T);
    double maxerr = 0.0;
    for(int t=0;t<5;t++){
        T hq[7],hqd[7],hu[7];
        for(int i=0;i<7;i++){ hq[i]=2.0f*rand()/RAND_MAX-1.0f; hqd[i]=2.0f*rand()/RAND_MAX-1.0f; hu[i]=2.0f*rand()/RAND_MAX-1.0f; }
        cudaMemcpy(d_q,hq,7*sizeof(T),cudaMemcpyHostToDevice);
        cudaMemcpy(d_qd,hqd,7*sizeof(T),cudaMemcpyHostToDevice);
        cudaMemcpy(d_u,hu,7*sizeof(T),cudaMemcpyHostToDevice);
        k_adapter<T><<<1,128,smem>>>(d_qa,d_q,d_qd,d_u,d_rm);
        k_ref<T><<<1,128,smem>>>(d_qr,d_q,d_qd,d_u,d_rm);
        cudaDeviceSynchronize();
        T ha[7],hr[7]; cudaMemcpy(ha,d_qa,7*sizeof(T),cudaMemcpyDeviceToHost); cudaMemcpy(hr,d_qr,7*sizeof(T),cudaMemcpyDeviceToHost);
        printf("t=%d adapter[0..2]=%.5f %.5f %.5f  ref=%.5f %.5f %.5f\n", t, ha[0],ha[1],ha[2], hr[0],hr[1],hr[2]);
        for(int i=0;i<7;i++){ double e=fabs((double)ha[i]-hr[i]); if(e>maxerr) maxerr=e; }
    }
    printf("MAX |qdd_adapter - qdd_ref| = %g  -> %s\n", maxerr, maxerr<1e-3?"PASS":"FAIL");
    return maxerr<1e-3?0:1;
}
