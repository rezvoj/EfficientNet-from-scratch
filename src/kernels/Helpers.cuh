
struct Multiply {
    __forceinline__ __device__
    float operator()(const float a, const float b) const { return a * b; }
};


struct Add {
    __forceinline__ __device__
    float operator()(const float a, const float b) const { return a + b; }
};

template <typename Operation>
static __global__ 
void elementwiseOpInplace(
        float* __restrict__ tensor, 
        const uint size) {
    const uint tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= size) return;
    const Operation operation;
    tensor[tIdx] = operation(tensor[tIdx]);
}


template <typename Operation>
static __global__
void elementwiseOp(
        float* __restrict__ outTensor,
        const float* __restrict__ inTensor,
        const uint size) {
    const uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) return;
    const Operation operation;
    outTensor[tid] = operation(inTensor[tid]);
}


// cublasSscal? (division, multiplication)
template <typename Operation>
static __global__ 
void elementwiseScalarOpInplace(
        float* __restrict__ tensor,
        const float value,
        const uint size) {
    const uint tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= size) return;
    const Operation operation;
    tensor[tIdx] = operation(tensor[tIdx], value);
}


template <typename Operation>
static __global__
void elementwiseScalarOp(
        float* __restrict__ outTensor,
        const float* __restrict__ inTensor,
        const float value,
        const uint size) {
    const uint tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= size) return;
    const Operation operation;
    outTensor[tIdx] = operation(inTensor[tIdx], value);
}


// cublasSaxpy ? (multiply and add)
template <typename Operation>
static __global__ 
void elementwise2TensorOpInplace(
        float* __restrict__ toTensor, 
        const float* __restrict__ fromTensor, 
        const uint size) {
    const uint tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= size) return;
    const Operation operation;
    toTensor[tIdx] = operation(toTensor[tIdx], fromTensor[tIdx]);
}


template <typename Operation>
static __global__ 
void elementwise2TensorOp(
        float* __restrict__ outTensor,
        const float* __restrict__ tensorA,
        const float* __restrict__ tensorB, 
        const uint size) {
    const uint tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= size) return;
    const Operation operation;
    outTensor[tIdx] = operation(tensorA[tIdx], tensorB[tIdx]);
}


// // Maybe usefull prolly not
// template <uint TensorDim>
// __global__
// void reshapeTensor(
//         float* __restrict__ tensorOut,
//         const float* __restrict__ tensorIn,
//         const uint outRowSize,
//         const uint outOrder[TensorDim],
//         const uint outDimSizeSums[TensorDim],
//         const uint inDimSizeSums[TensorDim]) {
//     const uint outFullIdx = blockIdx.x * blockDim.x + threadIdx.x;
//     uint outIdxes[TensorDim];
//     uint outCurrIdx = outFullIdx;
//     #pragma unroll
//     for (uint dimIdx = 0; dimIdx < TensorDim; ++dimIdx) {
//         const uint currentDimSizeSum = outDimSizeSums[dimIdx];
//         outIdxes[dimIdx] = outCurrIdx / currentDimSizeSum;
//         outCurrIdx = outCurrIdx % currentDimSizeSum;
//     }
//     float value = 0.0f;
//     if (outIdxes[TensorDim - 1] < outRowSize) {
//         uint inFullIdx = 0;
//         #pragma unroll
//         for (uint dimIdx = 0; dimIdx < TensorDim; ++dimIdx) {
//             const uint currDimIdx = outIdxes[outOrder[dimIdx]];
//             inFullIdx += currDimIdx * inDimSizeSums[currDimIdx];
//         }
//         value = tensorIn[inFullIdx];
//     }
//     tensorOut[outFullIdx] = value;
// }
