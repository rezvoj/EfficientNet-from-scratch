// src/layers/ConvolutionLayer.hpp

#pragma once
#include "Layer.hpp"

// Forward-declare the concrete class so we can use it in Model.hpp if needed
// without including the full template definition.
template<int FILTER_SIZE, int STRIDE>
class ConvolutionLayer;

// We templatize this layer on Filter Size and Stride
// to pass these values to the kernel launch.
template<int FILTER_SIZE, int STRIDE>
class ConvolutionLayer : public Layer {
public:
    // --- DECLARATIONS ONLY ---
    ConvolutionLayer(int in_channels, int out_channels);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& output_gradient) override;

    // Accessors for the optimizer
    Tensor* get_weights() { return &m_weights; }
    Tensor* get_weights_grad() { return &m_d_weights; }
    // TODO: Add biases and their gradients

private:
    int m_in_channels;
    int m_out_channels;

    Tensor m_weights;
    Tensor m_d_weights; // Gradient w.r.t weights

    // Caches for backward pass
    Tensor m_input_cache;
    Tensor m_col_matrix_cache; // Cache the im2col matrix for efficiency
};