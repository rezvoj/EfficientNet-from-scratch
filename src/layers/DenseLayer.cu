// src/layers/DenseLayer.cu

#include "DenseLayer.hpp"
#include "../kernels/Initialization.cuh" // <-- ADD THIS
#include "../kernels/DenseLayer.cuh"     // <-- ADD THIS
#include "../kernels/Helpers.cuh" // <-- ADD THIS LINE
#include <cuda_runtime.h>
#include <cmath>
#include <ctime>

// The constructor contains kernel launches, so its implementation belongs in a .cu file.
DenseLayer::DenseLayer(int input_size, int output_size)
    : m_weights(1, 1, output_size, input_size),
      m_biases(1, 1, 1, output_size),
      m_d_weights(1, 1, output_size, input_size),
      m_d_biases(1, 1, 1, output_size) {
    
    const uint threads = 256;
    const uint blocks_w = (m_weights.size() + threads - 1) / threads;
    float kaiming_range = sqrtf(2.0f / input_size);
    // Note: The 'true' is a template parameter, so it needs to be explicit
    initRandomValues<true><<<blocks_w, threads>>>(m_weights.data(), time(0), 0, kaiming_range, m_weights.size());
    CUDA_CHECK(cudaGetLastError());
    
    const uint blocks_b = (m_biases.size() + threads - 1) / threads;
    clearValue<<<blocks_b, threads>>>(m_biases.data(), 0.0f, m_biases.size());
    CUDA_CHECK(cudaGetLastError());
}

// The forward pass also contains kernel launches.
Tensor DenseLayer::forward(const Tensor& input) {
    m_input_cache = input;

    Tensor output(1, 1, input.H(), m_weights.H());
    gemm(input, m_weights, output, 1.0f, 0.0f, false, true);
    
    const uint threads = 256;
    const uint blocks = (output.size() + threads - 1) / threads;
    // Now the compiler knows what "Add" is because we included Helpers.cuh
    rowBroadcastOpInplace<Add><<<blocks, threads>>>(
        output.data(), m_biases.data(), m_biases.W(), output.size());
    CUDA_CHECK(cudaGetLastError());
    
    return output;
}

// The backward pass is pure C++ calling helper functions, but those helpers
// will eventually call CUDA, so we keep the whole implementation together.
Tensor DenseLayer::backward(const Tensor& output_gradient) {
    const int B = output_gradient.H();
    gemm(output_gradient, m_input_cache, m_d_weights, 1.0f, 0.0f, true, false);

    Tensor input_gradient(1, 1, B, m_weights.W());
    gemm(output_gradient, m_weights, input_gradient, 1.0f, 0.0f, false, false);
    
    return input_gradient;
}