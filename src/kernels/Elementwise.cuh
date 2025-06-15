#pragma once



__global__
void elementwiseAddInplace(
        float* __restrict__ outTensor,
        const float* __restrict__ tensor,
        const uint size) {
    const uint tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= size) return;
    outTensor[tIdx] += tensor[tIdx];
};



__global__
void elementwiseMul(
        float* __restrict__ outTensor,
        const float* __restrict__ tensorA,
        const float* __restrict__ tensorB,
        const uint size) {
    const uint tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= size) return;
    outTensor[tIdx] = tensorA[tIdx] * tensorB[tIdx];
};



__global__
void elementwiseMulInplace(
        float* __restrict__ tensorA,
        const float* __restrict__ inTensorB,
        const uint size) {
    const uint tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= size) return;
    tensorA[tIdx] *= inTensorB[tIdx];
};



__global__
void elementwiseMulBackwardInplace(
        float* __restrict__ tensorA,
        float* __restrict__ tensorB,
        const float* __restrict__ inGradTensor,
        const uint size) {
    const uint tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= size) return;
    const float valueA = tensorA[tIdx];
    const float valueB = tensorB[tIdx];
    const float ValueGrad = inGradTensor[tIdx];
    tensorA[tIdx] = ValueGrad * valueB;
    tensorB[tIdx] = ValueGrad * valueA;
};



__global__ 
void elementwiseScalarMulInplace(
        float* __restrict__ tensor,
        const float value,
        const uint size) {
    const uint tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= size) return;
    tensor[tIdx] *= value;
}
