static __global__ void sumOfSquares(const float* __restrict__ tensor, float* __restrict__ result, uint size) {
    extern __shared__ float sdata[];
    uint tid = threadIdx.x;
    uint i = blockIdx.x * (blockDim.x * 2) + tid;
    uint gridSize = blockDim.x * 2 * gridDim.x;

    float mySum = 0;
    while (i < size) {
        mySum += tensor[i] * tensor[i];
        if (i + blockDim.x < size) {
            mySum += tensor[i + blockDim.x] * tensor[i + blockDim.x];
        }
        i += gridSize;
    }
    sdata[tid] = mySum;
    __syncthreads();

    // Perform reduction in shared memory
    for (uint s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

// This kernel scales a tensor by a given factor if clipping is needed.
static __global__
void scaleTensor(float* __restrict__ tensor, float scale_factor, uint size) {
    const uint tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= size) return;
    tensor[tIdx] *= scale_factor;
}