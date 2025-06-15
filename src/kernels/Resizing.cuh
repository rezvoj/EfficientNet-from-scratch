#pragma once



__global__ 
void tensorExpansion(
        float* __restrict__ outTensor,
        const float* __restrict__ inTensor,
        const float scale,
        const uint fullSize,
        const uint outHWSize) {
    const uint tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= fullSize) return;
    const uint inIdx = tIdx / outHWSize;
    outTensor[tIdx] = scale * inTensor[inIdx];
}



__global__
void broadcastAddBiasInplace(
        float* __restrict__ outMatrix,
        const float* __restrict__ inBiases,
        const uint CSize,
        const uint size) {
    const uint tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= size) return;
    const uint biasIdx = tIdx % CSize;
    outMatrix[tIdx] += inBiases[biasIdx];
}
