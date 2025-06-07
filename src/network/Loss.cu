// src/network/Loss.cu

#include "Loss.hpp"
#include "../kernels/Loss.cuh"
#include "../kernels/Initialization.cuh"

float CrossEntropyLoss::forward(const Tensor& logits, const int* true_labels, int batch_size) {
    // Cache the label pointer and batch size for the backward pass
    m_labels_cache = true_labels;
    m_batch_size_cache = batch_size;

    const int NumClasses = logits.W();

    Tensor loss_gpu(1, 1, 1, 1);
    clearValue<<<1, 1>>>(loss_gpu.data(), 0.0f, 1);
    CUDA_CHECK(cudaGetLastError());

    m_probs_cache = Tensor(1, 1, batch_size, NumClasses);

    const uint threads_per_block = 256;
    dim3 gridDim(batch_size);
    dim3 blockDim(threads_per_block);
    size_t shared_mem_size = NumClasses * sizeof(float);

    softmaxCrossEntropyFwd<<<gridDim, blockDim, shared_mem_size>>>(
        loss_gpu.data(),
        m_probs_cache.data(),
        logits.data(),
        true_labels, // Pass the raw pointer directly
        batch_size,
        NumClasses
    );
    CUDA_CHECK(cudaGetLastError());
    
    float host_loss;
    CUDA_CHECK(cudaMemcpy(&host_loss, loss_gpu.data(), sizeof(float), cudaMemcpyDeviceToHost));
    
    return host_loss / batch_size;
}

Tensor CrossEntropyLoss::backward() {
    const int B = m_batch_size_cache;
    const int NumClasses = m_probs_cache.W();
    
    Tensor d_logits(1, 1, B, NumClasses);

    const uint threads_per_block = 256;
    dim3 gridDim((uint)(B * NumClasses + threads_per_block - 1) / threads_per_block);
    dim3 blockDim(threads_per_block);

    softmaxCrossEntropyBwd<<<gridDim, blockDim>>>(
        d_logits.data(),
        m_probs_cache.data(),
        m_labels_cache, // Use the cached raw pointer
        B,
        NumClasses
    );
    CUDA_CHECK(cudaGetLastError());

    return d_logits;
}