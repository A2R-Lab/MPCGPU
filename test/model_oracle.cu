// Export adapter dynamics/derivatives and generated kinematics for an independent
// Pinocchio oracle. Only correctness, no CUDA events or timing collection.
#include <cstdio>
#include <cmath>
#include "common/workspace.cuh"
#include "dynamics/rbd_plant.cuh"

__global__ void evaluate(float* output, grid::robotModel<float>* model, int sample) {
    extern __shared__ float scratch[];
    __shared__ float q[7], v[7], u[7], a[7], derivative[147], pose[6], jac[42], bias[7];
    for (int i=threadIdx.x;i<7;i+=blockDim.x) {
        q[i]=.4f*sinf(float(3*sample+i));
        v[i]=.2f*cosf(float(sample+2*i));
        u[i]=.5f*sinf(float(2*sample-i));
    }
    __syncthreads();
    mpcgpu_plant::forwardDynamicsAndGradient<float>(derivative,a,q,v,u,scratch,model);
    __syncthreads();
    grid::end_effector_pose_device_EE<float>(pose,q,model);
    __syncthreads();
    grid::end_effector_pose_gradient_device_EE<float>(jac,q,model);
    __syncthreads();
    grid::inverse_dynamics_device<float>(bias,q,v,model,nullptr,mpcgpu_plant::GRAVITY<float>());
    __syncthreads();
    if (threadIdx.x==0) {
        int k=0;
        for (auto p : {q,v,u,a}) for (int i=0;i<7;++i) output[k++]=p[i];
        for (int i=0;i<147;++i) output[k++]=derivative[i];
        for (int i=0;i<6;++i) output[k++]=pose[i];
        for (int i=0;i<42;++i) output[k++]=jac[i];
        for (int i=0;i<7;++i) output[k++]=bias[i];
    }
}

int main() try {
    auto* model=grid::init_robotModel<float>();
    constexpr int count=230;
    float *device, host[count];
    mpcgpu::checkCuda(cudaMalloc(&device,count*sizeof(float)));
    for (int sample=0;sample<5;++sample) {
        evaluate<<<1,128,24000>>>(device,model,sample);
        mpcgpu::checkCuda(cudaGetLastError());
        mpcgpu::checkCuda(cudaMemcpy(host,device,sizeof(host),cudaMemcpyDeviceToHost));
        printf("ORACLE");
        for (float value : host) { if (!std::isfinite(value)) return 1; printf(" %.9g",value); }
        printf("\n");
    }
    mpcgpu::checkCuda(cudaFree(device));
    mpcgpu::checkCuda(grid::free_robotModel_checked(model));
} catch (const std::exception& error) {
    fprintf(stderr,"%s\n",error.what()); return 1;
}
