// src/layers/Layer.hpp

#pragma once
#include "../utils/Tensor.hpp"
#include <memory>

// Abstract base class for all layers
class Layer {
public:
    virtual ~Layer() = default;

    // The forward pass takes a const reference to an input tensor
    // and returns a new tensor containing the output.
    virtual Tensor forward(const Tensor& input) = 0;

    // The backward pass takes the gradient of the loss with respect to the layer's output
    // and returns the gradient of the loss with respect to the layer's input.
    virtual Tensor backward(const Tensor& output_gradient) = 0;

    // A flag to switch between training and inference behavior (e.g., for BatchNorm)
    virtual void set_mode(bool is_training) {
        m_is_training = is_training;
    }

protected:
    bool m_is_training = true;
};