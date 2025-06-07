// src/layers/ConvolutionLayer.cu

#include "ConvolutionLayer.hpp"
#include "../utils/BlasUtils.hpp"
#include "../kernels/Convolutions.cuh"
#include "../kernels/Initialization.cuh"
#include <cmath> // For sqrtf
#include <ctime> // For time(0)

// --- Constructor ---
template<int FILTER_SIZE, int STRIDE>
ConvolutionLayer<FILTER_SIZE, STRIDE>::ConvolutionLayer(int in_channels, int out_channels)
    : m_in_channels(in_channels),
      m_out_channels(out_channels),
      // Weights are stored ready for GEMM: [C_out, C_in * F * F]
      m_weights(1, 1, out_channels, in_channels * FILTER_SIZE * FILTER_SIZE),
      m_d_weights(1, 1, out_channels, in_channels * FILTER_SIZE * FILTER_SIZE) {
    
    // Initialize weights using Kaiming He initialization
    const uint threads = 256;
    const uint blocks = (m_weights.size() + threads - 1) / threads;
    float kaiming_range = sqrtf(2.0f / (in_channels * FILTER_SIZE * FILTER_SIZE));
    initRandomValues<true><<<blocks, threads>>>(m_weights.data(), time(0), 1, kaiming_range, m_weights.size());
    CUDA_CHECK(cudaGetLastError());
}


// --- Forward Pass ---
template<int FILTER_SIZE, int STRIDE>
Tensor ConvolutionLayer<FILTER_SIZE, STRIDE>::forward(const Tensor& input) {
    const int B = input.B();
    const int H_in = input.H();
    const int W_in = input.W();
    
    const int H_out = (H_in - FILTER_SIZE) / STRIDE + 1; // Assuming 'valid' padding
    const int W_out = (W_in - FILTER_SIZE) / STRIDE + 1;

    m_input_cache = input;

    // 1. Create im2col matrix
    Tensor col_matrix(1, 1, m_in_channels * FILTER_SIZE * FILTER_SIZE, B * H_out * W_out);
    
    const uint threads = 256; // Using 256 is safer and often just as fast
    
    // Grid is 2D, which is correct for this kernel's logic.
    dim3 gridDim_im2col(
        (uint)(B * H_out * W_out + threads - 1) / threads,
        (uint)(m_in_channels * FILTER_SIZE * FILTER_SIZE) // The y-dim of the grid maps directly to this
    );

    // FIX: The block should be 1D. The kernel's internal logic uses threadIdx.x
    dim3 blockDim_im2col(threads);

    // The kernel itself is designed with 2D blockIdx, which is fine.
    (im2colConv<FILTER_SIZE, STRIDE>)<<<gridDim_im2col, blockDim_im2col>>>(
        col_matrix.data(), input.data(),
        m_in_channels, B, H_out * W_out, W_out, H_in, W_in
    );
    CUDA_CHECK(cudaGetLastError());

    // Cache this matrix for the dW calculation in the backward pass
    m_col_matrix_cache = col_matrix;

    // 2. Perform GEMM: Output = Weights * ColMatrix
    Tensor output_matrix(1, 1, m_out_channels, B * H_out * W_out);
    gemm(m_weights, col_matrix, output_matrix);

    // 3. Reshape output matrix to a 4D tensor
    output_matrix.reshape(B, m_out_channels, H_out, W_out);
    
    return output_matrix;
}


// --- Backward Pass ---
template<int FILTER_SIZE, int STRIDE>
Tensor ConvolutionLayer<FILTER_SIZE, STRIDE>::backward(const Tensor& output_gradient) {
    const int B = m_input_cache.B();
    const int H_in = m_input_cache.H();
    const int W_in = m_input_cache.W();
    const int H_out = output_gradient.H();
    const int W_out = output_gradient.W();

    // Reshape output_gradient from [B, C_out, H_out, W_out] to [C_out, B * H_out * W_out] for GEMM
    Tensor output_grad_matrix = output_gradient; // Shallow copy
    output_grad_matrix.reshape(1, 1, m_out_channels, B * H_out * W_out);

    // 1. Compute dLoss/dK (gradient for weights)
    // dW = dY * (im2col(X))^T
    gemm(output_grad_matrix, m_col_matrix_cache, m_d_weights, 1.0f, 0.0f, false, true);

    // 2. Compute dLoss/dX (gradient for input)
    // This requires a "col2im" operation. We do it via another GEMM.
    // First, calculate dY * W
    Tensor dcol(1, 1, m_in_channels * FILTER_SIZE * FILTER_SIZE, B * H_out * W_out);
    gemm(m_weights, output_grad_matrix, dcol, 1.0f, 0.0f, true, false);

    // TODO: The second part is the col2im kernel, which we haven't written yet.
    // It's the inverse of im2col. For now, we'll return a zero tensor.
    Tensor input_gradient(B, m_in_channels, H_in, W_in);
    const uint threads_init = 256;
    const uint blocks_init = (input_gradient.size() + threads_init - 1) / threads_init;
    clearValue<<<blocks_init, threads_init>>>(input_gradient.data(), 0.0f, input_gradient.size());

    return input_gradient;
}


// --- Explicit Template Instantiation ---
// We must tell the compiler which versions of this template to build.
// Add more instantiations as needed for your network (e.g., filter size 3, 5, stride 1, 2)
template class ConvolutionLayer<3, 1>;
template class ConvolutionLayer<3, 2>;
template class ConvolutionLayer<5, 1>;