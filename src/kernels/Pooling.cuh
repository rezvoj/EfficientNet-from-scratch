// src/kernels/Pooling.cuh

#pragma once
#include "../utils/CudaUtils.hpp"

// This kernel performs a global average pool.
// It takes a [B, C, H, W] tensor and reduces it to a [B, C, 1, 1] tensor
// by averaging over the H and W dimensions.
static __global__ void globalAveragePoolForward(
    float* __restrict__ out,      // Output tensor of shape [B, C, 1, 1]
    const float* __restrict__ in, // Input tensor of shape [B, C, H, W]
    int B, int C, int H, int W) {

    // Each block will process one C x B pair.
    int b = blockIdx.x;
    int c = blockIdx.y;

    float sum = 0.0f;
    // Iterate over all H and W for this specific batch and channel
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            size_t in_idx = (size_t)b * C * H * W + 
                            (size_t)c * H * W + 
                            (size_t)h * W + w;
            sum += in[in_idx];
        }
    }
    
    size_t out_idx = (size_t)b * C + c;
    out[out_idx] = sum / (H * W);
}

// The backward pass for global average pooling
static __global__ void globalAveragePoolBackward(
    float* __restrict__ d_in,
    const float* __restrict__ d_out,
    int B, int C, int H, int W) {

    // Each block handles one B, C pair.
    // Threads in the block handle the H, W plane.
    int b = blockIdx.x;
    int c = blockIdx.y;

    size_t d_out_idx = (size_t)b * C + c;
    float grad_val = d_out[d_out_idx] / (H * W);

    for (int i = threadIdx.x; i < H * W; i += blockDim.x) {
        int h = i / W;
        int w = i % W;

        size_t d_in_idx = (size_t)b * C * H * W + 
                          (size_t)c * H * W + 
                          (size_t)h * W + w;
        d_in[d_in_idx] = grad_val;
    }
}