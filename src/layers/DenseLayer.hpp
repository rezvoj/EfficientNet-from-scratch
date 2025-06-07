// src/layers/DenseLayer.hpp

#pragma once
#include "Layer.hpp"
#include "../utils/BlasUtils.hpp"
// DO NOT include kernel files here

class DenseLayer : public Layer {
public:
    // --- DECLARATIONS ONLY ---
    DenseLayer(int input_size, int output_size);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& output_gradient) override;

    // Accessors
    Tensor* get_weights() { return &m_weights; }
    Tensor* get_biases() { return &m_biases; }
    Tensor* get_weights_grad() { return &m_d_weights; }
    Tensor* get_biases_grad() { return &m_d_biases; }

private:
    Tensor m_weights;
    Tensor m_biases;
    Tensor m_d_weights;
    Tensor m_d_biases;
    Tensor m_input_cache;
};