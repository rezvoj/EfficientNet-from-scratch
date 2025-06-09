// src/network/Optimizer.cu

#include "Optimizer.hpp"
#include "../kernels/Optimizer.cuh"
#include "../kernels/Initialization.cuh"
#include "../kernels/Clip.cuh"
#include <cmath> // For sqrtf

AdamW::AdamW(Model* model, float learning_rate, float beta1, float beta2, float weight_decay, float epsilon)
    : m_lr(learning_rate),
      m_beta1(beta1),
      m_beta2(beta2),
      m_weight_decay(weight_decay),
      m_epsilon(epsilon),
      m_iteration(0),
      m_grad_norm_buffer(1, 1, 1, 1) { // Initialize the buffer for clipping

    // Get the parameters and gradients from the model
    m_params = model->get_parameters();
    m_grads = model->get_parameter_gradients();

    // Ensure the number of params and grads match
    assert(m_params.size() == m_grads.size());

    // Initialize the m and v moment tensors for each parameter
    for (const auto& param : m_params) {
        m_m_moments.emplace_back(param->B(), param->C(), param->H(), param->W());
        m_v_moments.emplace_back(param->B(), param->C(), param->H(), param->W());
    }

    // Zero out all the moment tensors
    const uint threads = 256;
    for (auto& moment : m_m_moments) {
        if (moment.size() == 0) continue;
        const uint blocks = (moment.size() + threads - 1) / threads;
        clearValue<<<blocks, threads>>>(moment.data(), 0.0f, moment.size());
    }
    for (auto& moment : m_v_moments) {
        if (moment.size() == 0) continue;
        const uint blocks = (moment.size() + threads - 1) / threads;
        clearValue<<<blocks, threads>>>(moment.data(), 0.0f, moment.size());
    }
    CUDA_CHECK(cudaGetLastError());
}

void AdamW::clip_gradients(float max_norm) {
    // This method calculates the total L2 norm of all gradient tensors combined
    // and scales them down if the norm exceeds max_norm.

    // Step 1: Calculate the sum of squares of all gradients
    clearValue<<<1, 1>>>(m_grad_norm_buffer.data(), 0.0f, 1);
    const uint threads = 256;

    for (const auto& grad_tensor : m_grads) {
        if (grad_tensor->size() == 0) continue;
        // The reduction kernel processes 2*blockDim elements per block
        const uint blocks = (grad_tensor->size() + (threads * 2) - 1) / (threads * 2);
        // Shared memory size is threads * sizeof(float)
        sumOfSquares<<<blocks, threads, threads * sizeof(float)>>>(
            grad_tensor->data(), m_grad_norm_buffer.data(), grad_tensor->size());
    }
    CUDA_CHECK(cudaGetLastError());

    // Step 2: Get the total norm on the CPU
    float sum_sq;
    CUDA_CHECK(cudaMemcpy(&sum_sq, m_grad_norm_buffer.data(), sizeof(float), cudaMemcpyDeviceToHost));
    float total_norm = sqrtf(sum_sq);

    // Step 3: If the norm exceeds the threshold, calculate the scaling factor and apply it
    if (total_norm > max_norm) {
        float scale_factor = max_norm / (total_norm + 1e-6); // Add epsilon to prevent division by zero

        for (auto& grad_tensor : m_grads) {
            if (grad_tensor->size() == 0) continue;
            const uint blocks = (grad_tensor->size() + threads - 1) / threads;
            scaleTensor<<<blocks, threads>>>(grad_tensor->data(), scale_factor, grad_tensor->size());
        }
        CUDA_CHECK(cudaGetLastError());
    }
}

void AdamW::step(int batch_size) {
    if (batch_size <= 0) {
        return;
    }

    m_iteration++;
    const uint threads = 256;

    // Calculate bias correction on the host
    const float bias_correction1 = 1.0f - powf(m_beta1, m_iteration);
    const float bias_correction2 = 1.0f - powf(m_beta2, m_iteration);

    for (size_t i = 0; i < m_params.size(); ++i) {
        Tensor* param = m_params[i];
        Tensor* grad = m_grads[i];
        Tensor* m_moment = &m_m_moments[i];
        Tensor* v_moment = &m_v_moments[i];

        if (param->size() == 0) continue;

        const uint blocks = (param->size() + threads - 1) / threads;

        adamWOptimizerStep<<<blocks, threads>>>(
            param->data(),
            m_moment->data(),
            v_moment->data(),
            grad->data(),
            bias_correction1,
            bias_correction2,
            m_lr,
            m_beta1,
            m_beta2,
            m_weight_decay,
            m_epsilon,
            1.0f, // batchSize is no longer used for division here
            param->size()
        );
    }
    CUDA_CHECK(cudaGetLastError());
}