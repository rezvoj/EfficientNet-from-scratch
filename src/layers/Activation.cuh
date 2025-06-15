#pragma once
#include <exception>
#include <random>
#include "Layer.cuh"
#include "../kernels/Activation.cuh"
#include "../utils/Exceptions.cuh"
#include "../utils/Math.cuh"

constexpr size_t BLOCK_SIZE = 256;



template <typename ACTIVATION>
class ActivationLayer : public Layer {
private:
    float* d_oSavedTensor;

public:
    ActivationLayer(const TensorSize inputSize): 
            Layer(inputSize, inputSize),
            d_oSavedTensor(nullptr) {}


    void toggleTrain(const bool trainOn) override {
        if (backOn == trainOn) return;
        backOn = trainOn;
        if (!trainOn) {
            // Free the buffer only needed during training
            checkCuda(cudaFree(d_oSavedTensor));
            d_oSavedTensor = nullptr;
        }
        // Reset the batch sizes to force buffer reallocations in forward pass
        currBatchSize = 0;
        currActualBatchSize = 0;
    }


    void forward(float* d_inputTensor, const size_t batchSize) override {
        const size_t fullSize = batchSize * inputSize.fullSize();
        // Save the batch size for backwards pass
        currBatchSize = batchSize;
        // Case for training
        if (backOn) {
            // Reallocate the buffer only if the actual size is smaller then batch size
            if (currActualBatchSize < batchSize) {
                currActualBatchSize = batchSize;
                checkCuda(cudaFree(d_oSavedTensor));
                checkCuda(cudaMalloc(&d_oSavedTensor, fullSize * sizeof(float)));
            }
            // Calculate the activation into the saved tensor
            elementwiseActivation<ACTIVATION>
                <<<ceilDiv(fullSize, BLOCK_SIZE), BLOCK_SIZE>>>(d_oSavedTensor, d_inputTensor, fullSize); 
            checkCudaLastError();
            // Claim the ownership of the input tensor in the place of the saved tensor
            std::swap(d_inputTensor, d_oSavedTensor);
        }
        // Case for inference
        else {
            // Calculate the activation inplace, since no need to save the original input data
            elementwiseActivationInplace<ACTIVATION>
                <<<ceilDiv(fullSize, BLOCK_SIZE), BLOCK_SIZE>>>(d_inputTensor, fullSize);
            checkCudaLastError();
        }
        // Lend the borrowed input tensor (or the swapped saved one) with the activation data
        next->forward(d_inputTensor, batchSize);
    }


    void backward(float* d_gradientTensor) override {
        const size_t fullSize = currBatchSize * inputSize.fullSize();
        // Calculate the gradient for prev layer inplace into the tensor given back by next layer
        elementwiseActivationBackwardInplace<ACTIVATION>
            <<<ceilDiv(fullSize, BLOCK_SIZE), BLOCK_SIZE>>>(d_gradientTensor, d_oSavedTensor, fullSize); 
        checkCudaLastError();
        // Complete memory the trade by giving back the recieved tensor with gradient data
        prev->backward(d_gradientTensor);
    }


    ~ActivationLayer() override {
        if (!std::uncaught_exceptions()) {
            checkCuda(cudaFree(d_oSavedTensor));
        }
    }

};
