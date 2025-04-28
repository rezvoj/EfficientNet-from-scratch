
// maybe can be fully avoided passing as transpose into matmul? 
__global__
void transposeMatrix(
        float* __restrict__ matrixOut,
        const float* __restrict__ matrixIn,
        const uint colSize,
        const uint rowSize) {
    const uint tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= colSize * rowSize) return;
    const uint colIdx = tIdx / rowSize;
    const uint rowIdx = tIdx % rowSize;
    matrixOut[tIdx] = matrixIn[rowIdx * colSize + colIdx];
}


template <typename Operation>
__global__
void rowBroadcastOpInplace(
        float* __restrict__ outValues,
        const float* __restrict__ inVec,
        const uint rowSize,
        const uint size) {
    const uint tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= size) return;
    const uint rowIdx = tIdx % rowSize;
    const Operation operation;
    outValues[tIdx] = operation(outValues[tIdx], inVec[rowIdx]);
}


template <typename Operation>
__global__
void rowBroadcastOp(
        float* __restrict__ outValues,
        const float* __restrict__ inValues,
        const float* __restrict__ inVec,
        const uint rowSize,
        const uint size) {
    const uint tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= size) return;
    const uint rowIdx = tIdx % rowSize;
    const Operation operation;
    outValues[tIdx] = operation(inValues[tIdx], inVec[rowIdx]);
}

// fused add / mul kernel etc. ?
