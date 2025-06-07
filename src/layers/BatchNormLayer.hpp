// src/layers/BatchNormLayer.hpp

#pragma once
#include "Layer.hpp"
#include "../utils/BlasUtils.hpp" // We'll need a reduction helper from here later
#include "../kernels/BatchNorm.cuh"
#include "../kernels/Reduction.cuh" // For sum reduction

class BatchNormLayer : public Layer {
public:
    BatchNormLayer(int num_channels, float momentum = 0.1f, float epsilon = 1e-5f)
        : m_num_channels(num_channels),
          m_momentum(momentum),
          m_epsilon(epsilon),
          m_gamma(1, num_channels, 1, 1),      // Trainable scale parameter
          m_beta(1, num_channels, 1, 1),       // Trainable shift parameter
          m_running_mean(1, num_channels, 1, 1), // Non-trainable state
          m_running_var(1, num_channels, 1, 1) { // Non-trainable state
        
        // TODO: Initialize gamma to 1s and beta to 0s
        //       Initialize running_mean to 0s and running_var to 1s
    }

    Tensor forward(const Tensor& input) override {
        assert(input.C() == m_num_channels);
        m_input_shape[0] = input.B();
        m_input_shape[1] = input.C();
        m_input_shape[2] = input.H();
        m_input_shape[3] = input.W();

        if (m_is_training) {
            return forward_train(input);
        } else {
            return forward_inference(input);
        }
    }

    Tensor backward(const Tensor& output_gradient) override {
        // TODO: Implement the backward pass
        // This will involve several kernel calls from BatchNorm.cuh
        return Tensor(); // Dummy return for now
    }

private:
    Tensor forward_train(const Tensor& input) {
        const int B = input.B();
        const int C = input.C();
        const int H = input.H();
        const int W = input.W();
        const size_t batch_size_prod = B * H * W;
        const size_t total_size = input.size();
        const uint threads = 256;

        // Step 1: Calculate batch mean. We need to sum over B, H, W for each channel.
        // This is a reduction. We need a temporary tensor for sums.
        // TODO: This requires a robust reduction implementation.
        // For now, we'll placeholder the logic.
        Tensor batch_mean(1, C, 1, 1);
        Tensor batch_var(1, C, 1, 1);
        
        // Placeholder for mean/var calculation.
        // A real implementation would use a highly optimized reduction kernel (e.g., from CUB).
        // Let's assume `batch_mean` and `batch_var` are filled.

        // Step 2: Update running statistics
        // out = (1-momentum)*out + momentum*in
        // We can do this with elementwise ops.

        // Step 3: Normalize the input
        m_input_cache = input; // Cache for backward pass
        Tensor output = input; // Create a copy to normalize in-place
        
        const uint blocks_norm = (total_size + threads - 1) / threads;
        normalizeTensorInplace<<<blocks_norm, threads>>>(
            output.data(), batch_mean.data(), batch_var.data(), m_epsilon, H * W, total_size
        );
        CUDA_CHECK(cudaGetLastError());
        
        // Cache the normalized output before scaling, needed for backward pass
        m_normalized_cache = output;

        // Step 4: Scale and shift (apply gamma and beta)
        const uint blocks_scale = (total_size + threads - 1) / threads;
        scaleShiftTensor<<<blocks_scale, threads>>>(
            output.data(), output.data(), m_gamma.data(), m_beta.data(), H*W, total_size
        );
        CUDA_CHECK(cudaGetLastError());

        return output;
    }

    Tensor forward_inference(const Tensor& input) {
        Tensor output = input; // Create a copy
        const size_t total_size = input.size();
        const uint threads = 256;
        const uint blocks = (total_size + threads - 1) / threads;

        // In inference, we use the saved running_mean and running_var
        normalizeScaleShiftTensorInplace<<<blocks, threads>>>(
            output.data(),
            m_running_mean.data(),
            m_running_var.data(),
            m_gamma.data(),
            m_beta.data(),
            m_epsilon,
            input.H() * input.W(),
            total_size
        );
        CUDA_CHECK(cudaGetLastError());

        return output;
    }

    // Parameters and state
    int m_num_channels;
    float m_momentum;
    float m_epsilon;
    Tensor m_gamma;
    Tensor m_beta;
    Tensor m_running_mean;
    Tensor m_running_var;

    // Caches for backward pass
    Tensor m_input_cache;
    Tensor m_normalized_cache;
    int m_input_shape[4];
};