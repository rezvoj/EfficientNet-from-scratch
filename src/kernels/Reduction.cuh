
__global__
void computeReductionOffsets(
        uint* __restrict__ outStartOffsets,
        uint* __restrict__ outEndOffsets,
        const uint rowSize,
        const uint size) {
    const uint tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= size) return;
    const uint startIdx = rowSize * tIdx;
    outStartOffsets[tIdx] = startIdx;
    outEndOffsets[tIdx] = startIdx + rowSize;
}


// maybe fuse with transpose
__global__
void broadcastValue2DimsUp(
        float* __restrict__ outTensor,
        const float* __restrict__ inTensor,
        const uint colSizeProd,
        const uint size) {
    const uint tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= size) return;
    outTensor[tIdx] = inTensor[tIdx / colSizeProd];
}
