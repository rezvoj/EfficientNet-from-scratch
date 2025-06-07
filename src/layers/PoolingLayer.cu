// src/layers/PoolingLayer.cu
#include "PoolingLayer.hpp"
#include "../kernels/Pooling.cuh" // <-- Include the kernel definitions here

Tensor GlobalAveragePoolLayer::forward(const Tensor& input) {
    m_input_shape[0] = input.B();
    m_input_shape[1] = input.C();
    m_input_shape[2] = input.H();
    m_input_shape[3] = input.W();

    Tensor output(m_input_shape[0], m_input_shape[1], 1, 1);
    dim3 gridDim(m_input_shape[0], m_input_shape[1]);
    dim3 blockDim(256); // Use a 1D block now
    globalAveragePoolForward<<<gridDim, 1>>>(
        output.data(), input.data(),
        input.B(), input.C(), input.H(), input.W()
    );
    CUDA_CHECK(cudaGetLastError());
    return output;
}

Tensor GlobalAveragePoolLayer::backward(const Tensor& output_gradient) {
    Tensor input_gradient(m_input_shape[0], m_input_shape[1], m_input_shape[2], m_input_shape[3]);
    dim3 gridDim(m_input_shape[0], m_input_shape[1]);
    
    // Use a fixed block size, kernel must check bounds
    dim3 blockDim(32, 32); 
    // Note: I made a mistake in the previous kernel, it should be 1D for H and W, let's fix that.
    // It's better to make the kernel simpler.
    
    globalAveragePoolBackward<<<gridDim, blockDim>>>(
        input_gradient.data(), output_gradient.data(),
        m_input_shape[0], m_input_shape[1], m_input_shape[2], m_input_shape[3]
    );
    CUDA_CHECK(cudaGetLastError());
    return input_gradient;
}