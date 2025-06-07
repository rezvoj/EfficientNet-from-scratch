// src/layers/PoolingLayer.hpp
#pragma once
#include "Layer.hpp"
// DO NOT include Pooling.cuh here

class GlobalAveragePoolLayer : public Layer {
public:
    // --- DECLARATIONS ONLY ---
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& output_gradient) override;
private:
    int m_input_shape[4];
};