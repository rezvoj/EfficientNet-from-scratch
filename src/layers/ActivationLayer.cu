// src/layers/ActivationLayer.cu

#include "ActivationLayer.hpp"
#include "../kernels/Activations.cuh"
#include "../kernels/Helpers.cuh"
#include "../utils/CudaUtils.hpp"

// --- Template Implementation ---

template <typename ForwardOp, typename BackwardOp>
Tensor ActivationLayer<ForwardOp, BackwardOp>::forward(const Tensor& input) {
    m_output_cache = input;
    
    const uint num_elements = m_output_cache.size();
    if (num_elements > 0) {
        const uint threads_per_block = 256;
        const uint blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;
        
        elementwiseOpInplace<ForwardOp><<<blocks_per_grid, threads_per_block>>>(m_output_cache.data(), num_elements);
        CUDA_CHECK(cudaGetLastError());
    }
    
    return std::move(m_output_cache);
}

template <typename ForwardOp, typename BackwardOp>
Tensor ActivationLayer<ForwardOp, BackwardOp>::backward(const Tensor& output_gradient) {
    Tensor derivative = m_output_cache;
    const uint num_elements = derivative.size();
    if (num_elements > 0) {
        const uint threads_per_block = 256;
        const uint blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;

        elementwiseOpInplace<BackwardOp><<<blocks_per_grid, threads_per_block>>>(derivative.data(), num_elements);
        CUDA_CHECK(cudaGetLastError());
    }

    Tensor input_gradient = output_gradient;
    if (num_elements > 0) {
        const uint threads_per_block = 256;
        const uint blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;
        
        elementwise2TensorOpInplace<Multiply><<<blocks_per_grid, threads_per_block>>>(
            input_gradient.data(), derivative.data(), num_elements);
        CUDA_CHECK(cudaGetLastError());
    }

    return std::move(input_gradient);
}


// --- Explicit Template Instantiation ---
// This tells the nvcc compiler to generate the actual code for these specific
// versions of the template, so the linker can find them later.
template class ActivationLayer<ReLU, dReLU>;
template class ActivationLayer<SiLU, dSiLU>;