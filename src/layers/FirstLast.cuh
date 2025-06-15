#pragma once
#include <exception>
#include <cudnn_v9.h>
#include "Layer.cuh"
#include "../kernels/CrossEntropy.cuh"
#include "../utils/Exceptions.cuh"
#include "../utils/Math.cuh"

constexpr uint BLOCK_SIZE = 256;



class SoftmaxLossLayer : public Layer {
private:
    float epsilon;
    uint* d_oLabelValues;
    float* d_oLossValues;
    float* d_bPrevTensor;
    cudnnHandle_t cudnnHandle;
    cudnnTensorDescriptor_t cudnnTensorDesc;

public:
    SoftmaxLossLayer(
            const TensorSize inputSize,
            const cudnnHandle_t handle,
            const float epsilon):
                Layer(inputSize, inputSize),
                epsilon(epsilon),
                d_oLabelValues(nullptr),
                d_bPrevTensor(nullptr),
                cudnnHandle(handle) {
        checkCudnn(cudnnCreateTensorDescriptor(&cudnnTensorDesc));
    }


    void forward(float* d_inputTensor, const uint batchSize) override {
        // Save the borrowed input tensor for backprop
        d_bPrevTensor = d_inputTensor;
        // Conditionally change the tensor descriptors on batch size change
        if (currBatchSize != batchSize) {
            currBatchSize = batchSize;
            checkCudnn(cudnnSetTensor4dDescriptor(
                cudnnTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                batchSize, inputSize.C, 1, 1
            ));
            // Reallocate memory only if it's actual size is smaller
            if (currActualBatchSize < batchSize) {
                currActualBatchSize = batchSize;
                checkCuda(cudaFree(d_oLabelValues));
                checkCuda(cudaFree(d_oLossValues));
                checkCuda(cudaMalloc(&d_oLabelValues, batchSize * sizeof(uint)));
                checkCuda(cudaMalloc(&d_oLossValues, batchSize * sizeof(float)));
            }
        }
        // Compute the probabilities inplace from logits
        const float alpha = 1.0f;
        const float beta = 0.0f;
        checkCudnn(cudnnSoftmaxForward(
            cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
            &alpha, cudnnTensorDesc, d_inputTensor,
            &beta, cudnnTensorDesc, d_inputTensor
        ));
    }

    
    // Use start backward with host labels instead of backward
    void backward(float* d_gradientTensor) override {}


    // Fills the host buffer with the calculated probabilities
    // Only valid before starting backprop
    void getHostProbs(float* probTensor) {
        const uint copySizeBytes = currBatchSize * inputSize.C * sizeof(float);
        checkCuda(cudaMemcpy(probTensor, d_bPrevTensor, copySizeBytes, cudaMemcpyDeviceToHost));
    }


    // Fills the host buffer with the calculated probabilities
    // Only valid before starting backprop
    // Deeply assumes that the maximal label is less then number of categories
    void getHostBatchLoss(float* batchLoss, uint* trueLabels) {
        // Copy labels to device
        checkCuda(cudaMemcpy(
            d_oLabelValues, trueLabels, 
            currBatchSize * sizeof(uint),
            cudaMemcpyHostToDevice
        ));
        // Calculate cross entropy loss into device loss values
        crossEntropyLoss<<<ceilDiv(currBatchSize, BLOCK_SIZE), BLOCK_SIZE>>>(
            d_oLossValues, d_bPrevTensor, d_oLabelValues,
            epsilon, currBatchSize, inputSize.C
        );
        checkCudaLastError();
        // Copy cross entropy loss values to host
        checkCuda(cudaMemcpy(
            batchLoss, d_oLossValues, 
            currBatchSize * sizeof(float),
            cudaMemcpyDeviceToHost
        ));
    }


    // Starts the backprop chain taking and calculates the batch corss entropy loss
    void startBackward(uint* trueLabels) {
        if (!backOn) return;
        // Copy labels to device
        checkCuda(cudaMemcpy(
            d_oLabelValues, trueLabels, 
            currBatchSize * sizeof(uint),
            cudaMemcpyHostToDevice
        ));
        // Calculate loss gradients into the borrowed input tensor
        const uint fullSize = currBatchSize * inputSize.C;
        crossEntropyLossGradInplace<<<ceilDiv(fullSize, BLOCK_SIZE), BLOCK_SIZE>>>(
            d_bPrevTensor, d_oLabelValues, currBatchSize, inputSize.C
        );
        checkCudaLastError();
        // Initiate the packprop by returning the borrowed tensor with loss greadient data
        prev->backward(d_bPrevTensor);
    }


    ~SoftmaxLossLayer() override {
        checkCuda(cudaFree(d_oLabelValues));
        checkCuda(cudaFree(d_oLossValues));
        checkCudnn(cudnnDestroyTensorDescriptor(cudnnTensorDesc));
    }

};



class InputLayer : public Layer {
private:
    float* d_oOutTensor;
    
public:
    InputLayer(const TensorSize outputSize):
            Layer(outputSize, outputSize),
            d_oOutTensor(nullptr) {}

    
    // Use start forward with host inputs instead of forward
    void forward(float* d_inputTensor, const uint batchSize) override {}
    

    // Copies the host input tensor to device and starts the forward pass chain
    void startForward(float* inputTensor, const uint batchSize) {
        const uint fullSize = batchSize * outputSize.fullSize();
        // Reallocate memory only if it's actual size is smaller
        if (currActualBatchSize < batchSize) {
            currActualBatchSize = batchSize;
            checkCuda(cudaFree(d_oOutTensor));
            checkCuda(cudaMalloc(&d_oOutTensor, fullSize * sizeof(float)));
        }
        // Copy the input host tensor into the layer's owned output tensor
        checkCuda(cudaMemcpy(
            d_oOutTensor, inputTensor, 
            fullSize * sizeof(float),
            cudaMemcpyHostToDevice
        ));
        // Lend the output tensor with the loaded host data to the next layer
        next->forward(d_oOutTensor, batchSize);
    }


    void backward(float* d_gradientTensor) override {
        // Reclaim the given back output tensor buffer to complete the memory trade
        d_oOutTensor = d_gradientTensor;
    }


    ~InputLayer() override {
        if (!std::uncaught_exceptions()) {
            checkCuda(cudaFree(d_oOutTensor));
        }
    }

};
