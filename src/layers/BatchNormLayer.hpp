// src/layers/BatchNormLayer.hpp

#pragma once
#include "Layer.hpp"

class BatchNormLayer : public Layer {
public:
    // --- DECLARATIONS ONLY ---
    BatchNormLayer(int num_channels, float momentum = 0.1f, float epsilon = 1e-5f);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& output_gradient) override;

    // Accessors for the optimizer
    Tensor* get_gamma() { return &m_gamma; }
    Tensor* get_beta() { return &m_beta; }
    Tensor* get_gamma_grad() { return &m_d_gamma; }
    Tensor* get_beta_grad() { return &m_d_beta; }

private:
    Tensor forward_train(const Tensor& input);
    Tensor forward_inference(const Tensor& input);

    // Parameters and state
    int m_num_channels;
    float m_momentum;
    float m_epsilon;
    Tensor m_gamma;        // Trainable scale parameter
    Tensor m_beta;         // Trainable shift parameter
    Tensor m_d_gamma;      // Gradient for gamma
    Tensor m_d_beta;       // Gradient for beta
    Tensor m_running_mean; // Non-trainable state for inference
    Tensor m_running_var;  // Non-trainable state for inference

    // Caches for backward pass
    Tensor m_input_cache;
    Tensor m_normalized_cache;
    Tensor m_batch_mean_cache;
    Tensor m_batch_var_cache;
};