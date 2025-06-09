// src/layers/FlattenLayer.hpp

#pragma once
#include "Layer.hpp"

class FlattenLayer : public Layer {
public:
    Tensor forward(const Tensor& input) override {
        // Cache the original shape for the backward pass
        m_input_shape[0] = input.B();
        m_input_shape[1] = input.C();
        m_input_shape[2] = input.H();
        m_input_shape[3] = input.W();

        // Create a non-owning view and reshape it
        Tensor output = input; // This is a shallow copy (just pointers)
        output.reshape(1, 1, m_input_shape[0], m_input_shape[1] * m_input_shape[2] * m_input_shape[3]);
        return output;
    }

    Tensor backward(const Tensor& output_gradient) override {
        // The backward pass of flatten is just a reshape back to the original shape
        Tensor input_gradient = output_gradient; // Shallow copy
        input_gradient.reshape(m_input_shape[0], m_input_shape[1], m_input_shape[2], m_input_shape[3]);
        return input_gradient;
    }
private:
    int m_input_shape[4];
};