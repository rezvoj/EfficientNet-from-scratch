#pragma once
#include <curand_kernel.h>



template <bool UseNormal>
__global__
void initRandomValues(
        float* __restrict__ outValues,
        const uint seed,
        const uint offset,
        const float range,
        const uint size) {
    const uint tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= size) return;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, tIdx, offset, &state);
    float randVal;
    if constexpr (UseNormal) randVal = curand_normal(&state);
    else randVal = curand_uniform(&state)* 2.0f - 1.0f;
    outValues[tIdx] = randVal * range;
}


__global__
void initDropoutMask(
        float* __restrict__ outMask,
        const uint seed,
        const uint offset,
        const float dropRate,
        const uint size) {
    const uint tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= size) return;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, tIdx, offset, &state);
    const float randVal = curand_uniform(&state);
    outMask[tIdx] = randVal > dropRate ? 1.0f / (1.0f - dropRate) : 0.0f;
}


__global__
void clearValue(
        float* __restrict__ outValues,
        const float value,
        const uint size) {
    const uint tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= size) return;
    outValues[tIdx] = value;
}
