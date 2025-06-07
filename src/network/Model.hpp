// src/network/Model.hpp

#pragma once
#include "../layers/Layer.hpp"
#include "../layers/DenseLayer.hpp" // We need the concrete type to cast to
#include <vector>
#include <memory>

class Model {
public:
    // Add a layer to the network
    void add(std::unique_ptr<Layer> layer) {
        m_layers.push_back(std::move(layer));
    }

    // The full forward pass for the entire network
    Tensor forward(const Tensor& input) {
        Tensor current_output = input;
        for (const auto& layer : m_layers) {
            current_output = layer->forward(current_output);
        }
        return current_output;
    }

    // The full backward pass for the entire network
    void backward(const Tensor& loss_gradient) {
        Tensor current_gradient = loss_gradient;
        // Iterate through the layers in reverse order
        for (auto it = m_layers.rbegin(); it != m_layers.rend(); ++it) {
            current_gradient = (*it)->backward(current_gradient);
        }
    }

    // Set the mode for all layers (training vs. inference)
    void set_mode(bool is_training) {
        for (const auto& layer : m_layers) {
            layer->set_mode(is_training);
        }
    }

    // Collect all trainable parameters for the optimizer
    std::vector<Tensor*> get_parameters() {
        std::vector<Tensor*> params;
        for (const auto& layer : m_layers) {
            // Use dynamic_cast to check if the layer is of a trainable type
            if (auto dense_layer = dynamic_cast<DenseLayer*>(layer.get())) {
                params.push_back(dense_layer->get_weights());
                params.push_back(dense_layer->get_biases());
            }
            // TODO: Add similar checks for ConvolutionLayer, BatchNormLayer, etc.
        }
        return params;
    }

    std::vector<Tensor*> get_parameter_gradients() {
        std::vector<Tensor*> grads;
        for (const auto& layer : m_layers) {
            if (auto dense_layer = dynamic_cast<DenseLayer*>(layer.get())) {
                grads.push_back(dense_layer->get_weights_grad());
                grads.push_back(dense_layer->get_biases_grad());
            }
            // TODO: Add similar checks for other trainable layers
        }
        return grads;
    }

private:
    std::vector<std::unique_ptr<Layer>> m_layers;
};