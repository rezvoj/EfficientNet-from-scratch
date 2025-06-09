// src/network/Model.cu

#include "Model.hpp"
// We need the full definitions of the layers here
#include "../layers/DenseLayer.hpp"
#include "../layers/ConvolutionLayer.hpp"
#include "../layers/BatchNormLayer.hpp"
#include <fstream>
#include <vector>
#include <iostream>

void Model::add(std::unique_ptr<Layer> layer) {
    m_layers.push_back(std::move(layer));
}

Tensor Model::forward(const Tensor& input) {
    Tensor current_output = input;
    // Note: To pass a non-owning view, we must use std::move if we re-assign.
    // A simple copy is safer here as the tensor is small.
    for (const auto& layer : m_layers) {
        current_output = layer->forward(current_output);
    }
    return current_output;
}

void Model::backward(const Tensor& loss_gradient) {
    Tensor current_gradient = loss_gradient;
    for (auto it = m_layers.rbegin(); it != m_layers.rend(); ++it) {
        current_gradient = (*it)->backward(current_gradient);
    }
}

void Model::set_mode(bool is_training) {
    for (const auto& layer : m_layers) {
        layer->set_mode(is_training);
    }
}

void Model::save(const std::string& filepath) {
    std::cout << "Saving model parameters to " << filepath << "..." << std::endl;
    
    std::ofstream out_file(filepath, std::ios::binary);
    if (!out_file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filepath);
    }

    std::vector<Tensor*> params = this->get_parameters();

    for (const auto& param : params) {
        size_t num_elements = param->size();
        if (num_elements == 0) continue;

        std::vector<float> host_buffer(num_elements);
        
        CUDA_CHECK(cudaMemcpy(
            host_buffer.data(),
            param->data(),
            num_elements * sizeof(float),
            cudaMemcpyDeviceToHost
        ));

        out_file.write(
            reinterpret_cast<const char*>(host_buffer.data()),
            num_elements * sizeof(float)
        );
    }

    out_file.close();
    std::cout << "Model saved successfully." << std::endl;
}

void Model::load(const std::string& filepath) {
    std::cout << "Loading model parameters from " << filepath << "..." << std::endl;

    std::ifstream in_file(filepath, std::ios::binary);
    if (!in_file.is_open()) {
        throw std::runtime_error("Cannot open file for reading: " + filepath);
    }

    std::vector<Tensor*> params = this->get_parameters();

    for (auto& param : params) {
        size_t num_elements = param->size();
        if (num_elements == 0) continue;

        std::vector<float> host_buffer(num_elements);

        in_file.read(
            reinterpret_cast<char*>(host_buffer.data()),
            num_elements * sizeof(float)
        );

        if (in_file.gcount() != (long long)num_elements * sizeof(float)) {
            throw std::runtime_error("Error reading weights from file. File may be corrupt or mismatched.");
        }

        CUDA_CHECK(cudaMemcpy(
            param->data(),
            host_buffer.data(),
            num_elements * sizeof(float),
            cudaMemcpyHostToDevice
        ));
    }

    in_file.close();
    std::cout << "Model loaded successfully." << std::endl;
}

// Kernel to find the index of the maximum value in each row (argmax)
__global__ void argmaxKernel(
    int* __restrict__ predictions,      // Output: [B]
    const float* __restrict__ logits,  // Input: [B, NumClasses]
    int B, int NumClasses) {
    
    int b = blockIdx.x;
    if (b >= B) return;

    float max_val = -FLT_MAX;
    int max_idx = -1;

    for (int i = 0; i < NumClasses; ++i) {
        float val = logits[b * NumClasses + i];
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }
    predictions[b] = max_idx;
}

std::vector<int> Model::predict(const Tensor& input) {
    // 1. Perform a forward pass to get the logits
    Tensor logits = this->forward(input);
    const int B = logits.H();
    const int NumClasses = logits.W();

    // 2. Allocate GPU memory for the predictions (class indices)
    int* d_predictions;
    CUDA_CHECK(cudaMalloc(&d_predictions, B * sizeof(int)));

    // 3. Launch the argmax kernel
    argmaxKernel<<<B, 1>>>(d_predictions, logits.data(), B, NumClasses);
    CUDA_CHECK(cudaGetLastError());

    // 4. Copy the predicted indices from GPU to CPU
    std::vector<int> host_predictions(B);
    CUDA_CHECK(cudaMemcpy(
        host_predictions.data(),
        d_predictions,
        B * sizeof(int),
        cudaMemcpyDeviceToHost
    ));

    // 5. Free the temporary GPU memory
    CUDA_CHECK(cudaFree(d_predictions));

    return host_predictions;
}

// Collect all trainable parameters for the optimizer
std::vector<Tensor*> Model::get_parameters() {
    std::vector<Tensor*> params;
    for (const auto& layer : m_layers) {
        if (auto* dense_layer = dynamic_cast<DenseLayer*>(layer.get())) {
            params.push_back(dense_layer->get_weights());
        }
        else if (auto* bn_layer = dynamic_cast<BatchNormLayer*>(layer.get())) {
            params.push_back(bn_layer->get_gamma());
            params.push_back(bn_layer->get_beta());
        }
        else if (auto* conv_layer_3_1 = dynamic_cast<ConvolutionLayer<3, 1>*>(layer.get())) {
            params.push_back(conv_layer_3_1->get_weights());
        }
        else if (auto* bn_layer = dynamic_cast<BatchNormLayer*>(layer.get())) {
            params.push_back(bn_layer->get_gamma());
            params.push_back(bn_layer->get_beta());
        }
        // TODO: Add similar checks for BatchNormLayer, etc.
    }
    return params;
}

std::vector<Tensor*> Model::get_parameter_gradients() {
    std::vector<Tensor*> grads;
    for (const auto& layer : m_layers) {
        if (auto* dense_layer = dynamic_cast<DenseLayer*>(layer.get())) {
            grads.push_back(dense_layer->get_weights_grad());
        }
        else if (auto* bn_layer = dynamic_cast<BatchNormLayer*>(layer.get())) {
            grads.push_back(bn_layer->get_gamma_grad());
            grads.push_back(bn_layer->get_beta_grad());
        }
        else if (auto* conv_layer_3_1 = dynamic_cast<ConvolutionLayer<3, 1>*>(layer.get())) {
            grads.push_back(conv_layer_3_1->get_weights_grad());
        }
        // The second, buggy BatchNormLayer block is now removed.
    }
    return grads;
}