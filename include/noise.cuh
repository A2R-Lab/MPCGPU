#pragma once
#include <curand.h>
#include <curand_kernel.h>
#include <cstdint>
#include "settings.cuh"



// if ADD_NOISE is enabled a value from a rand_normal(mean=0, std_dev=1)*NOISE_MULTIPLIER will be added to a current joint state every NOISE_FREQUENCY control updates

template <typename T>
__global__
void addNoiseKernel(uint32_t state_size, T *d_x, T freq, T q_factor, T qd_factor, unsigned long long seed){
    curandState_t state;
    curand_init(seed, threadIdx.x, 0, &state);
    for(int ind = threadIdx.x; ind < state_size; ind+=blockDim.x){
        if (curand_uniform(&state) < freq){
            d_x[ind] += curand_normal(&state) * (ind < state_size / 2) * q_factor + (ind >= state_size/2) * qd_factor;
        }
    }
}

template <typename T>
void addNoise(uint32_t state_size, T *d_x, T frequency, T q_factor, T qd_factor){
    const unsigned long long seed = 12345;
    addNoiseKernel<<<1,32>>>(state_size, d_x, frequency, q_factor, qd_factor, seed);
}