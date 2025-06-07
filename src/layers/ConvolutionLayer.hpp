// src/layers/ConvolutionLayer.hpp

#pragma once
#include "Layer.hpp"
#include "../utils/BlasUtils.hpp"
#include "../kernels/Convolutions.cuh"

// We need to templatize this layer on Filter Size and Stride
// to pass these values to the kernel launch.
template<int FILTER_SIZE, int STRIDE>
class ConvolutionLayer : public Layer {
public:
    ConvolutionLayer(int in_channels, int out_channels)
        : m_in_channels(in_channels),
          m_out_channels(out_channels),
          m_weights(1, 1, out_channels, in_channels * FILTER_SIZE * FILTER_SIZE) {
        // The weights tensor is stored ready for GEMM: [C_out, C_in * H_f * W_f]
        
        // TODO: Add bias tensor and initialize weights/biases
    }

    Tensor forward(const Tensor& input) override {
        const int B = input.B();
        const int H_in = input.H();
        const int W_in = input.W();
        
        // Calculate output dimensions
        const int H_out = (H_in - FILTER_SIZE) / STRIDE + 1;
        const int W_out = (W_in - FILTER_SIZE) / STRIDE + 1;

        // Cache input for backward pass
        m_input_cache = input;

        // 1. Create im2col matrix
        // Shape: [C_in * F * F, B * H_out * W_out]
        Tensor col_matrix(1, 1, m_in_channels * FILTER_SIZE * FILTER_SIZE, B * H_out * W_out);

        const int out_hw_size = H_out * W_out;
        const int threads = 512;
        dim3 gridDim(
            (uint)(B * out_hw_size + threads - 1) / threads,
            (uint)(m_in_channels * FILTER_SIZE * FILTER_SIZE + threads - 1) / threads
        );
        dim3 blockDim(threads, threads);

        im2colConv<FILTER_SIZE, STRIDE><<<gridDim, blockDim>>>(
            col_matrix.data(), input.data(),
            m_in_channels, B, out_hw_size, W_out, H_in, W_in
        );
        CUDA_CHECK(cudaGetLastError());

        // 2. Perform GEMM
        // Output = Weights * ColMatrix
        // [C_out, B*H_out*W_out] = [C_out, C_in*F*F] @ [C_in*F*F, B*H_out*W_out]
        Tensor output_matrix(1, 1, m_out_channels, B * H_out * W_out);
        gemm(m_weights, col_matrix, output_matrix);

        // 3. Reshape output matrix to a 4D tensor
        // This is a "free" operation as we just reinterpret the pointer with new dimensions
        // by creating a new Tensor via the move constructor.
         // 3. Reshape output matrix to a 4D tensor
        Tensor output_tensor = std::move(output_matrix); // Take ownership of the pointer
        output_tensor.reshape(B, m_out_channels, H_out, W_out);

        // TODO: Add bias

        return output_tensor;

    }

    Tensor backward(const Tensor& output_gradient) override {
        // TODO: Implement backward pass
        return Tensor(); // Dummy
    }

private:
    int m_in_channels;
    int m_out_channels;
    Tensor m_weights;
    Tensor m_input_cache;
};