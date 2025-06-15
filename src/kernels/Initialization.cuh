#pragma once
#include <curand_kernel.h>



__global__
void initValues(
        float* __restrict__ outValues,
        const float value,
        const uint size) {
    const uint tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= size) return;
    outValues[tIdx] = value;
}



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
    else randVal = curand_uniform(&state) * 2.0f - 1.0f;
    outValues[tIdx] = randVal * range;
}
