#pragma once
#include <exception>
#include <cudnn_v9.h>
#include "Layer.cuh"
#include "../kernels/Resizing.cuh"
#include "../utils/Exceptions.cuh"
#include "../utils/Math.cuh"

constexpr uint BLOCK_SIZE = 256;



class AvgPoolingFlattenLayer : public Layer {
private:
    float* d_oOutTensor;
    float* d_bPrevTensor;
    cudnnHandle_t cudnnHandle;
    cudnnPoolingDescriptor_t cudnnPoolingDesc;
    cudnnTensorDescriptor_t cudnnInTensorDesc;
    cudnnTensorDescriptor_t cudnnOutTensorDesc;

public:
    AvgPoolingFlattenLayer(const TensorSize inputSize, const cudnnHandle_t handle):
            Layer(inputSize, {inputSize.C, 1, 1}),
            d_oOutTensor(nullptr),
            d_bPrevTensor(nullptr),
            cudnnHandle(handle) {
        checkCudnn(cudnnCreatePoolingDescriptor(&cudnnPoolingDesc));
        checkCudnn(cudnnCreateTensorDescriptor(&cudnnInTensorDesc));
        checkCudnn(cudnnCreateTensorDescriptor(&cudnnOutTensorDesc));
        checkCudnn(cudnnSetPooling2dDescriptor(
            cudnnPoolingDesc, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
            CUDNN_NOT_PROPAGATE_NAN, inputSize.H, inputSize.W, 0, 0, 1, 1
        ));
    }


    void forward(float* d_inputTensor, const uint batchSize) override {
        // Save the borrowed input tensor
        d_bPrevTensor = d_inputTensor;
        // Conditionally change the tensor descriptors on batch size change
        if (currBatchSize != batchSize) {
            currBatchSize = batchSize;
            checkCudnn(cudnnSetTensor4dDescriptor(
                cudnnInTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                batchSize, inputSize.C, inputSize.H, inputSize.W
            ));
            checkCudnn(cudnnSetTensor4dDescriptor(
                cudnnOutTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                batchSize, outputSize.C, 1, 1
            ));
            // Reallocate memory only if it's actual size is smaller
            if (currActualBatchSize < batchSize) {
                currActualBatchSize = batchSize;
                checkCuda(cudaFree(d_oOutTensor));
                checkCuda(cudaMalloc(&d_oOutTensor, batchSize * outputSize.fullSize() * sizeof(float)));
            }
        }
        // Flatten the input tensor into the output tensor
        const float alpha = 1.0f;
        const float beta = 0.0f;
        checkCudnn(cudnnPoolingForward(
            cudnnHandle, cudnnPoolingDesc,
            &alpha, cudnnInTensorDesc, d_bPrevTensor,
            &beta, cudnnOutTensorDesc, d_oOutTensor
        ));
        // Lend the output tensor with activation data to the next layer
        next->forward(d_oOutTensor, batchSize);
    }


    void backward(float* d_gradientTensor) override {
        const uint fullSize = currBatchSize * inputSize.fullSize();
        // Reclaim the incoming gradient tensor as owned output tensor
        d_oOutTensor = d_gradientTensor;
        // Expand and rescale the output tensor into the input tensor
        const float scale = 1.0f / static_cast<float>(inputSize.H * inputSize.W);
        tensorExpansion<<<ceilDiv(fullSize, BLOCK_SIZE), BLOCK_SIZE>>>(
            d_bPrevTensor, d_oOutTensor, scale, fullSize, inputSize.H * inputSize.W
        );
        checkCudaLastError();
        // Complete memory the trade by giving back the saved input tensor with gradient data
        prev->backward(d_bPrevTensor);
    }


    ~AvgPoolingFlattenLayer() override {
        if (!std::uncaught_exceptions()) {
            checkCuda(cudaFree(d_oOutTensor));
            checkCudnn(cudnnDestroyPoolingDescriptor(cudnnPoolingDesc));
            checkCudnn(cudnnDestroyTensorDescriptor(cudnnInTensorDesc));
            checkCudnn(cudnnDestroyTensorDescriptor(cudnnOutTensorDesc));
        }
    }

};



class ExpansionLayer : public Layer {
private:    
    float* d_oOutTensor;
    float* d_bPrevTensor;
    cudnnHandle_t cudnnHandle;
    cudnnReduceTensorDescriptor_t cudnnReduceDesc;
    cudnnTensorDescriptor_t cudnnInTensorDesc;
    cudnnTensorDescriptor_t cudnnOutTensorDesc;

public:
    ExpansionLayer(const TensorSize outputSize, const cudnnHandle_t handle):
            Layer({outputSize.C, 1, 1}, outputSize),
            d_oOutTensor(nullptr),
            d_bPrevTensor(nullptr),
            cudnnHandle(handle) {
        checkCudnn(cudnnCreateReduceTensorDescriptor(&cudnnReduceDesc));
        checkCudnn(cudnnCreateTensorDescriptor(&cudnnInTensorDesc));
        checkCudnn(cudnnCreateTensorDescriptor(&cudnnOutTensorDesc));
        checkCudnn(cudnnSetReduceTensorDescriptor(
            cudnnReduceDesc, CUDNN_REDUCE_TENSOR_ADD, CUDNN_DATA_FLOAT,
            CUDNN_NOT_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES
        ));
    }


    void forward(float* d_inputTensor, const uint batchSize) override {
        const uint fullSize = batchSize * outputSize.fullSize();
        // Save the borrowed input tensor
        d_bPrevTensor = d_inputTensor;
        // Conditionally change the tensor descriptors on batch size change
        if (currBatchSize != batchSize) {
            currBatchSize = batchSize;
            checkCudnn(cudnnSetTensor4dDescriptor(
                cudnnInTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                batchSize, outputSize.C, 1, 1
            ));
            checkCudnn(cudnnSetTensor4dDescriptor(
                cudnnOutTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                batchSize, outputSize.C, outputSize.H, outputSize.W
            ));
            // Reallocate memory only if it's actual size is smaller
            if (currActualBatchSize < batchSize) {
                currActualBatchSize = batchSize;
                checkCuda(cudaFree(d_oOutTensor));
                checkCuda(cudaMalloc(&d_oOutTensor, fullSize * sizeof(float)));
            }
        }
        // Expand the input tensor into the output tensor
        tensorExpansion<<<ceilDiv(fullSize, BLOCK_SIZE), BLOCK_SIZE>>>(
            d_oOutTensor, d_bPrevTensor, 1.0f, fullSize, outputSize.H * outputSize.W
        );
        checkCudaLastError();
        // Lend the output tensor with activation data to the next layer
        next->forward(d_oOutTensor, batchSize);
    }


    void backward(float* d_gradientTensor) override {
        // Reclaim the incoming gradient tensor as owned output tensor
        d_oOutTensor = d_gradientTensor;
        // Reduce the output tensor into the input tensor
        const float alpha = 1.0f;
        const float beta = 0.0f;
        checkCudnn(cudnnReduceTensor(
            cudnnHandle, cudnnReduceDesc,
            nullptr, 0, nullptr, 0,
            &alpha, cudnnOutTensorDesc, d_oOutTensor,
            &beta, cudnnInTensorDesc, d_bPrevTensor
        ));
        // Complete memory the trade by giving back the saved input tensor with gradient data
        prev->backward(d_bPrevTensor);
    }


    ~ExpansionLayer() override {
        if (!std::uncaught_exceptions()) {
            checkCudnn(cudnnDestroyReduceTensorDescriptor(cudnnReduceDesc));
            checkCudnn(cudnnDestroyTensorDescriptor(cudnnInTensorDesc));
            checkCudnn(cudnnDestroyTensorDescriptor(cudnnOutTensorDesc));
            checkCuda(cudaFree(d_oOutTensor));
        }
    }

};
