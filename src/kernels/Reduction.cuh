
static __global__ void computeReductionOffsets(
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
static __global__
void broadcastValue2DimsUp(
        float* __restrict__ outTensor,
        const float* __restrict__ inTensor,
        const uint colSizeProd,
        const uint size) {
    const uint tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= size) return;
    outTensor[tIdx] = inTensor[tIdx / colSizeProd];
}


// Reduces a [B, C, H, W] tensor to a [C] vector by summing over B, H, W.
static __global__ void reduceSum(
    float* __restrict__ out_vec,       // Output: [C]
    const float* __restrict__ in_tensor, // Input: [B, C, H, W]
    int B, int C, int H, int W) {
    
    // Each block processes one channel
    int c = blockIdx.x;
    if (c >= C) return;

    float sum = 0.0f;
    // Each thread sums up a portion of the B, H, W dimensions for this channel
    for (int i = threadIdx.x; i < B * H * W; i += blockDim.x) {
        int b = i / (H * W);
        int h = (i / W) % H;
        int w = i % W;

        size_t idx = (size_t)b * C * H * W + (size_t)c * H * W + (size_t)h * W + w;
        sum += in_tensor[idx];
    }

    // A simple atomic add reduction. For higher performance, a shared memory reduction is better.
    atomicAdd(&out_vec[c], sum);
}