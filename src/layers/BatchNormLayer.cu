// src/layers/BatchNormLayer.cu

#include "BatchNormLayer.hpp"
#include "../kernels/Initialization.cuh"
#include "../kernels/Reduction.cuh"
#include "../kernels/BatchNorm.cuh"
#include "../kernels/Helpers.cuh"
#include <vector>

// Constructor to initialize parameters and state
BatchNormLayer::BatchNormLayer(int num_channels, float momentum, float epsilon)
    : m_num_channels(num_channels),
      m_momentum(momentum),
      m_epsilon(epsilon),
      m_gamma(1, 1, 1, num_channels),
      m_beta(1, 1, 1, num_channels),
      m_d_gamma(1, 1, 1, num_channels),
      m_d_beta(1, 1, 1, num_channels),
      m_running_mean(1, 1, 1, num_channels),
      m_running_var(1, 1, 1, num_channels) {
    
    const uint threads = 256;
    const uint blocks_params = (num_channels + threads - 1) / threads;

    clearValue<<<blocks_params, threads>>>(m_gamma.data(), 1.0f, m_gamma.size());
    clearValue<<<blocks_params, threads>>>(m_beta.data(), 0.0f, m_beta.size());
    clearValue<<<blocks_params, threads>>>(m_running_mean.data(), 0.0f, m_running_mean.size());
    clearValue<<<blocks_params, threads>>>(m_running_var.data(), 1.0f, m_running_var.size());
    CUDA_CHECK(cudaGetLastError());
}

// Main forward method, switches between train and inference
Tensor BatchNormLayer::forward(const Tensor& input) {
    assert(input.C() == m_num_channels);
    if (m_is_training) {
        return forward_train(input);
    } else {
        return forward_inference(input);
    }
}

// Forward pass during training
Tensor BatchNormLayer::forward_train(const Tensor& input) {
    const int B = input.B();
    const int C = input.C();
    const int H = input.H();
    const int W = input.W();
    const size_t spatial_dim = H * W;
    const size_t samples_per_channel = B * spatial_dim;
    const size_t total_size = input.size();
    const uint threads = 256;
    const uint blocks_c = (C + threads - 1) / threads;
    const uint blocks_total = (total_size + threads - 1) / threads;

    m_input_cache = input;

    // --- Step 1 & 2: Calculate batch mean and variance ---
    Tensor batch_mean(1, 1, 1, C);
    clearValue<<<C, 1>>>(batch_mean.data(), 0.0f, C);
    reduceSum<<<C, threads>>>(batch_mean.data(), input.data(), B, C, H, W);
    (elementwiseScalarOpInplace<Divide>)<<<blocks_c, threads>>>(batch_mean.data(), (float)samples_per_channel, C);

    Tensor diff_sq(B, C, H, W);
    computeMeanDiffSquared<<<blocks_total, threads>>>(diff_sq.data(), input.data(), batch_mean.data(), spatial_dim, total_size);

    Tensor batch_var(1, 1, 1, C);
    clearValue<<<C, 1>>>(batch_var.data(), 0.0f, C);
    reduceSum<<<C, threads>>>(batch_var.data(), diff_sq.data(), B, C, H, W);
    (elementwiseScalarOpInplace<Divide>)<<<blocks_c, threads>>>(batch_var.data(), (float)samples_per_channel, C);
    
    // --- NEW, CRITICAL FIX ---
    // Clamp the variance to be >= 0 to prevent sqrt(negative) -> NaN
    floor_op<<<blocks_c, threads>>>(batch_var.data(), 0.0f, C);

    m_batch_mean_cache = batch_mean;
    m_batch_var_cache = batch_var;

    // --- FIX: Step 3: Update running statistics correctly ---
    // Create new, temporary owning tensors for the scaled components.
    Tensor scaled_running_mean = m_running_mean; // Deep copy
    Tensor scaled_batch_mean = batch_mean;       // Deep copy
    
    (elementwiseScalarOpInplace<Multiply>)<<<blocks_c, threads>>>(scaled_running_mean.data(), 1.0f - m_momentum, C);
    (elementwiseScalarOpInplace<Multiply>)<<<blocks_c, threads>>>(scaled_batch_mean.data(), m_momentum, C);
    
    // Now perform the final addition into the destination tensor m_running_mean
    (elementwise2TensorOp<Add>)<<<blocks_c, threads>>>(m_running_mean.data(), scaled_running_mean.data(), scaled_batch_mean.data(), C);

    // Repeat for variance
    Tensor scaled_running_var = m_running_var; // Deep copy
    Tensor scaled_batch_var = batch_var;       // Deep copy

    (elementwiseScalarOpInplace<Multiply>)<<<blocks_c, threads>>>(scaled_running_var.data(), 1.0f - m_momentum, C);
    (elementwiseScalarOpInplace<Multiply>)<<<blocks_c, threads>>>(scaled_batch_var.data(), m_momentum, C);

    (elementwise2TensorOp<Add>)<<<blocks_c, threads>>>(m_running_var.data(), scaled_running_var.data(), scaled_batch_var.data(), C);

    // --- Step 4: Normalize, Scale, and Shift ---
    Tensor output = input;
    m_normalized_cache = input;

    // During training, always normalize with the current batch's stats.
    normalizeTensorInplace<<<blocks_total, threads>>>(
        m_normalized_cache.data(), batch_mean.data(), batch_var.data(), m_epsilon, spatial_dim, total_size);
    
    scaleShiftTensor<<<blocks_total, threads>>>(
        output.data(), m_normalized_cache.data(), m_gamma.data(), m_beta.data(), spatial_dim, total_size
    );
    CUDA_CHECK(cudaGetLastError());
    return output;
}

// Forward pass during inference
Tensor BatchNormLayer::forward_inference(const Tensor& input) {
    Tensor output = input;
    const size_t total_size = input.size();
    const size_t spatial_dim = input.H() * input.W();
    const uint threads = 256;
    const uint blocks = (total_size + threads - 1) / threads;

    normalizeScaleShiftTensorInplace<<<blocks, threads>>>(
        output.data(), m_running_mean.data(), m_running_var.data(),
        m_gamma.data(), m_beta.data(), m_epsilon, spatial_dim, total_size
    );
    CUDA_CHECK(cudaGetLastError());
    return output;
}

// Backward pass
Tensor BatchNormLayer::backward(const Tensor& output_gradient) {
    const int B = m_input_cache.B();
    const int C = m_input_cache.C();
    const int H = m_input_cache.H();
    const int W = m_input_cache.W();
    const size_t spatial_dim = H * W;
    const size_t total_size = m_input_cache.size();
    const uint threads = 256;
    const uint blocks_c = (C + threads - 1) / threads;
    const uint blocks_total = (total_size + threads - 1) / threads;

    // --- Step 1: Calculate gradients for gamma and beta ---
    clearValue<<<blocks_c, threads>>>(m_d_gamma.data(), 0.0f, C);
    clearValue<<<blocks_c, threads>>>(m_d_beta.data(), 0.0f, C);

    // d_gamma = sum(d_out * normalized_input) over B, H, W
    Tensor temp_mul(B, C, H, W);
    (elementwise2TensorOp<Multiply>)<<<blocks_total, threads>>>(
        temp_mul.data(), output_gradient.data(), m_normalized_cache.data(), total_size);
    reduceSum<<<C, threads>>>(m_d_gamma.data(), temp_mul.data(), B, C, H, W);
    
    // d_beta = sum(d_out) over B, H, W
    reduceSum<<<C, threads>>>(m_d_beta.data(), output_gradient.data(), B, C, H, W);

    // --- Step 2: Calculate gradient for the input (dLoss/dX) ---
    // This is the most complex step, using the chain rule.
    // We start with d_out and modify it in-place using the batchNormPrevGradInplace kernel.
    Tensor input_gradient = output_gradient; // Creates a deep copy to be modified.

    batchNormPrevGradInplace<<<blocks_total, threads>>>(
        input_gradient.data(),
        m_gamma.data(),
        m_batch_var_cache.data(),
        m_normalized_cache.data(),
        m_d_beta.data(),
        m_d_gamma.data(),
        m_epsilon,
        spatial_dim,
        total_size
    );
    CUDA_CHECK(cudaGetLastError());

    return input_gradient;
}