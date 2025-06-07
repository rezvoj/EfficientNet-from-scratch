// src/network/Optimizer.cu

#include "Optimizer.hpp"
#include "../kernels/Optimizer.cuh"
#include "../kernels/Initialization.cuh"

AdamW::AdamW(Model* model, float learning_rate, float beta1, float beta2, float weight_decay, float epsilon)
    : m_lr(learning_rate),
      m_beta1(beta1),
      m_beta2(beta2),
      m_weight_decay(weight_decay),
      m_epsilon(epsilon),
      m_iteration(0) {

    // Get the parameters and gradients from the model
    m_params = model->get_parameters();
    m_grads = model->get_parameter_gradients();

    // Initialize the m and v moment tensors for each parameter
    for (const auto& param : m_params) {
        m_m_moments.emplace_back(param->B(), param->C(), param->H(), param->W());
        m_v_moments.emplace_back(param->B(), param->C(), param->H(), param->W());
    }

    // Zero out all the moment tensors
    const uint threads = 256;
    for (auto& moment : m_m_moments) {
        const uint blocks = (moment.size() + threads - 1) / threads;
        clearValue<<<blocks, threads>>>(moment.data(), 0.0f, moment.size());
    }
    for (auto& moment : m_v_moments) {
        const uint blocks = (moment.size() + threads - 1) / threads;
        clearValue<<<blocks, threads>>>(moment.data(), 0.0f, moment.size());
    }
    CUDA_CHECK(cudaGetLastError());
}

void AdamW::step() {
    m_iteration++;
    const uint threads = 256;

    // Loop through all parameters and apply the update step
    for (size_t i = 0; i < m_params.size(); ++i) {
        Tensor* param = m_params[i];
        Tensor* grad = m_grads[i];
        Tensor* m_moment = &m_m_moments[i];
        Tensor* v_moment = &m_v_moments[i];

        const uint blocks = (param->size() + threads - 1) / threads;

        adamWOptimizerStep<<<blocks, threads>>>(
            param->data(),
            m_moment->data(),
            v_moment->data(),
            grad->data(),
            m_iteration,
            m_lr,
            m_beta1,
            m_beta2,
            m_weight_decay,
            m_epsilon,
            1.0f, // Using a batch size of 1 for gradient update, as grad is already averaged
            param->size()
        );
    }
    CUDA_CHECK(cudaGetLastError());
}
