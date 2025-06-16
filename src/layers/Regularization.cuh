#pragma once
#include <exception>
#include <cudnn_v9.h>
#include "Layer.cuh"
#include "../kernels/Initialization.cuh"
#include "../network/Optimizer.cuh"
#include "../utils/Exceptions.cuh"
#include "../utils/Math.cuh"

constexpr uint BLOCK_SIZE = 256;



class DropoutLayer : public Layer {
private:
    float dropoutRate;
    uint dropoutSeed;
    bool skipInputGrad;
    size_t cudnnDropoutStatesSize;
    size_t cudnnReserveSpaceSize;
    void* d_oCudnnDropoutStates;
    void* d_oCudnnReserveSpace;
    cudnnHandle_t cudnnHandle;
    cudnnDropoutDescriptor_t cudnnDropoutDesc;
    cudnnTensorDescriptor_t cudnnTensorDesc;

public:
    DropoutLayer(
            const TensorSize inputSize,
            const float rate,
            const uint seed,
            const bool skipInputGrad,
            const cudnnHandle_t handle):
                Layer(inputSize, inputSize),
                dropoutRate(rate),
                dropoutSeed(seed),
                skipInputGrad(skipInputGrad),
                cudnnDropoutStatesSize(0),
                cudnnReserveSpaceSize(0),
                d_oCudnnDropoutStates(nullptr),
                d_oCudnnReserveSpace(nullptr),
                cudnnHandle(handle) {
        checkCudnn(cudnnCreateDropoutDescriptor(&cudnnDropoutDesc));
        checkCudnn(cudnnDropoutGetStatesSize(cudnnHandle, &cudnnDropoutStatesSize));
        checkCudnn(cudnnCreateTensorDescriptor(&cudnnTensorDesc));
    }


    void toggleTrain(const bool trainOn) override {
        if (backOn == trainOn) return;
        backOn = trainOn;
        if (trainOn) {
            // Create the dropout states for training if not created yet
            if (d_oCudnnDropoutStates == nullptr) {
                checkCuda(cudaMalloc(&d_oCudnnDropoutStates, cudnnDropoutStatesSize));
                checkCudnn(cudnnSetDropoutDescriptor(
                    cudnnDropoutDesc, cudnnHandle, dropoutRate,
                    d_oCudnnDropoutStates, cudnnDropoutStatesSize, dropoutSeed
                ));
            }
        }
        else {
            // Free the rosources needed only for training
            checkCuda(cudaFree(d_oCudnnReserveSpace));
            d_oCudnnReserveSpace = nullptr;
        }
        // Reset the batch sizes to force buffer reallocations in forward pass
        cudnnReserveSpaceSize = 0;
        currBatchSize = 0;
        currActualBatchSize = 0;
    }


    void forward(float* d_inputTensor, const uint batchSize) override {
        if (backOn) {
            // Conditionally change the tensor descriptor on batch size change
            if (currBatchSize != batchSize) {
                currBatchSize = batchSize;
                checkCudnn(cudnnSetTensor4dDescriptor(
                    cudnnTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                    batchSize, inputSize.C, inputSize.H, inputSize.W
                ));
                // Reallocate memory only if it's actual size is smaller
                if (currActualBatchSize < batchSize) {
                    currActualBatchSize = batchSize;
                    checkCuda(cudaFree(d_oCudnnReserveSpace));
                    checkCudnn(cudnnDropoutGetReserveSpaceSize(cudnnTensorDesc, &cudnnReserveSpaceSize));
                    checkCuda(cudaMalloc(&d_oCudnnReserveSpace, cudnnReserveSpaceSize));
                }
            }
            // Compute the dropout forward pass inplace
            checkCudnn(cudnnDropoutForward(
                cudnnHandle, cudnnDropoutDesc,
                cudnnTensorDesc, d_inputTensor,
                cudnnTensorDesc, d_inputTensor,   
                d_oCudnnReserveSpace, cudnnReserveSpaceSize
            ));
        }
        // Lend the borrowed tensor to the next layer
        next->forward(d_inputTensor, batchSize);
    }

    
    void backward(float* d_gradientTensor) override {
        // Calculate the backward pass on the returned tensor
        if (!skipInputGrad) {
            checkCudnn(cudnnDropoutBackward(
                cudnnHandle, cudnnDropoutDesc,
                cudnnTensorDesc, d_gradientTensor,
                cudnnTensorDesc, d_gradientTensor,
                d_oCudnnReserveSpace, cudnnReserveSpaceSize
            ));
        }
        // Complete memory the trade by returning the input tensor
        //  conditionally with gradient data
        prev->backward(d_gradientTensor);
    }


    ~DropoutLayer() override {
        if (!std::uncaught_exceptions()) {
            checkCuda(cudaFree(d_oCudnnDropoutStates));
            checkCuda(cudaFree(d_oCudnnReserveSpace));
            checkCudnn(cudnnDestroyDropoutDescriptor(cudnnDropoutDesc));
            checkCudnn(cudnnDestroyTensorDescriptor(cudnnTensorDesc));
        }
    }

};



class BatchNormLayer : public LearnableLayer {
private:
    float expAvgFactor;
    float epsilon;
    float *d_oOutTensor, *d_oBackOutTensor;
    float *d_oScale, *d_oShift;
    float *d_oRunningMean, *d_oRunningVar;
    float *d_oScaleGrads, *d_oShiftGrads;
    float *d_oBatchMean, *d_oBatchInvVariance;
    float* d_bPrevTensor;
    cudnnHandle_t cudnnHandle;
    cudnnTensorDescriptor_t cudnnFullTensorDesc;
    cudnnTensorDescriptor_t cudnnChannelFlatTensorDesc;

public:
    BatchNormLayer(
            const TensorSize inputSize,
            const float expAvgFactor,
            const float epsilon,
            const Optimizer::OptimAlgo algorithm,
            const cudnnHandle_t handle):
                LearnableLayer(inputSize, inputSize, algorithm),
                expAvgFactor(expAvgFactor),
                epsilon(epsilon),
                d_oOutTensor(nullptr),
                d_oBackOutTensor(nullptr),
                d_oScale(nullptr), d_oShift(nullptr),
                d_oRunningMean(nullptr), d_oRunningVar(nullptr),
                d_oScaleGrads(nullptr), d_oShiftGrads(nullptr),
                d_oBatchMean(nullptr), d_oBatchInvVariance(nullptr),
                d_bPrevTensor(nullptr),
                cudnnHandle(handle) {
        checkCudnn(cudnnCreateTensorDescriptor(&cudnnFullTensorDesc));
        checkCudnn(cudnnCreateTensorDescriptor(&cudnnChannelFlatTensorDesc));
        checkCudnn(cudnnSetTensor4dDescriptor(
            cudnnChannelFlatTensorDesc, 
            CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
            1, inputSize.C, 1, 1
        ));
        checkCuda(cudaMalloc(&d_oScale, outputSize.C * sizeof(float)));
        checkCuda(cudaMalloc(&d_oShift, outputSize.C * sizeof(float)));
        checkCuda(cudaMalloc(&d_oRunningMean, outputSize.C * sizeof(float)));
        checkCuda(cudaMalloc(&d_oRunningVar, outputSize.C * sizeof(float)));
    }


    void toggleTrain(const bool trainOn) override {
        if (backOn == trainOn) return;
        backOn = trainOn;
        if (trainOn) {
            // Allocate resources independant on batch size
            checkCuda(cudaMalloc(&d_oScaleGrads, outputSize.C * sizeof(float)));
            checkCuda(cudaMalloc(&d_oShiftGrads, outputSize.C * sizeof(float)));
            checkCuda(cudaMalloc(&d_oBatchMean, outputSize.C * sizeof(float)));
            checkCuda(cudaMalloc(&d_oBatchInvVariance, outputSize.C * sizeof(float)));
        }
        else {
            // Free the buffers only needed during training
            checkCuda(cudaFree(d_oBackOutTensor));
            checkCuda(cudaFree(d_oScaleGrads));
            checkCuda(cudaFree(d_oShiftGrads));
            checkCuda(cudaFree(d_oBatchMean));
            checkCuda(cudaFree(d_oBatchInvVariance));
            d_oBackOutTensor = nullptr;
            d_oScaleGrads = nullptr;
            d_oShiftGrads = nullptr;
            d_oBatchMean = nullptr;
            d_oBatchInvVariance = nullptr;
        }
        // Reset the batch sizes to force buffer reallocations in forward pass
        currBatchSize = 0;
        currActualBatchSize = 0;
    }


    void initWeights([[maybe_unused]] const uint seed) override {
        // Initialize the scale, shift, running mean and variance to default values
        const dim3 gridSize(ceilDiv(outputSize.C, BLOCK_SIZE));
        initValues<<<gridSize, BLOCK_SIZE>>>(d_oScale, 1.0f, outputSize.C);
        checkCudaLastError();
        initValues<<<gridSize, BLOCK_SIZE>>>(d_oShift, 0.0f, outputSize.C);
        checkCudaLastError();
        initValues<<<gridSize, BLOCK_SIZE>>>(d_oRunningMean, 0.0f, outputSize.C);
        checkCudaLastError();
        initValues<<<gridSize, BLOCK_SIZE>>>(d_oRunningVar, 1.0f, outputSize.C);
        checkCudaLastError();
    }


    void registerWeights(Optimizer& optimizer) override {
        if (backOn) {
            optimizer.registerLayer(optimizerAlgorithm, d_oScale, d_oScaleGrads, outputSize.C);
            optimizer.registerLayer(optimizerAlgorithm, d_oShift, d_oShiftGrads, outputSize.C);
        }
    }


    void reRegisterGrads(Optimizer& optimizer) override {
        if (backOn) {
            optimizer.reRegisterLayerGrads(optimizerAlgorithm, d_oScale, d_oScaleGrads);
            optimizer.reRegisterLayerGrads(optimizerAlgorithm, d_oShift, d_oShiftGrads);
        }
    }


    void forward(float* d_inputTensor, const uint batchSize) override {
        // Save the borrowed input tensor for backward pass
        d_bPrevTensor = d_inputTensor;
        // Conditionally change the tensor descriptors on batch size change
        if (currBatchSize != batchSize) {
            currBatchSize = batchSize;
            checkCudnn(cudnnSetTensor4dDescriptor(
                cudnnFullTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                batchSize, outputSize.C, outputSize.H, outputSize.W
            ));
            // Reallocate memory only if it's actual size is smaller
            if (currActualBatchSize < batchSize) {
                currActualBatchSize = batchSize;
                const uint fullSizeBytes = batchSize * outputSize.fullSize() * sizeof(float);
                checkCuda(cudaFree(d_oOutTensor));
                checkCuda(cudaMalloc(&d_oOutTensor, fullSizeBytes));
                // Reallocate the backprob output tensor only during training 
                if (backOn) {
                    checkCuda(cudaFree(d_oBackOutTensor));
                    checkCuda(cudaMalloc(&d_oBackOutTensor, fullSizeBytes));
                }
            }
        }
        const float alpha = 1.0f;
        const float beta = 0.0f;
        // Compute the forward pass activations during training using batch moments
        if (backOn) {
            checkCudnn(cudnnBatchNormalizationForwardTraining(
                cudnnHandle, CUDNN_BATCHNORM_SPATIAL, &alpha, &beta,
                cudnnFullTensorDesc, d_inputTensor,
                cudnnFullTensorDesc, d_oOutTensor,
                cudnnChannelFlatTensorDesc,
                d_oScale, d_oShift,
                expAvgFactor, d_oRunningMean, d_oRunningVar,
                epsilon, d_oBatchMean, d_oBatchInvVariance
            ));
        }
        // Compute the forward pass activations during inference using learned moments
        else {
            checkCudnn(cudnnBatchNormalizationForwardInference(
                cudnnHandle, CUDNN_BATCHNORM_SPATIAL, &alpha, &beta,
                cudnnFullTensorDesc, d_inputTensor,
                cudnnFullTensorDesc, d_oOutTensor,
                cudnnChannelFlatTensorDesc,
                d_oScale, d_oShift,
                d_oRunningMean, d_oRunningVar,
                epsilon
            ));
        }
        // Lend the output tensor with activation data to the next layer
        next->forward(d_oOutTensor, batchSize);
    }


    void backward(float* d_gradientTensor) override {
        // Reclaim the incoming gradient tensor as owned output tensor
        d_oOutTensor = d_gradientTensor;
        // Accumulate the the parameter gradients and compute input gradients
        const float alpha = 1.0f;
        const float beta_overwrite = 0.0f;
        const float beta_accumulate = 1.0f;
        checkCudnn(cudnnBatchNormalizationBackward(
            cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
            &alpha, &beta_overwrite, &alpha, &beta_accumulate,
            cudnnFullTensorDesc, d_bPrevTensor,
            cudnnFullTensorDesc, d_gradientTensor,
            cudnnFullTensorDesc, d_oBackOutTensor,
            cudnnChannelFlatTensorDesc,
            d_oScale, d_oScaleGrads, d_oShiftGrads,
            epsilon, d_oBatchMean, d_oBatchInvVariance
        ));
        // Complete memory the trade by trading the owned backward output tensor
        //  with gradient data for saved input tensor from previous layer
        prev->backward(d_oBackOutTensor);
        d_oBackOutTensor = d_bPrevTensor;
    }


    ~BatchNormLayer() override {
        if (!std::uncaught_exceptions()) {
            checkCuda(cudaFree(d_oOutTensor));
            checkCuda(cudaFree(d_oBackOutTensor));
            checkCuda(cudaFree(d_oScale));
            checkCuda(cudaFree(d_oShift));
            checkCuda(cudaFree(d_oRunningMean));
            checkCuda(cudaFree(d_oRunningVar));
            checkCuda(cudaFree(d_oScaleGrads));
            checkCuda(cudaFree(d_oShiftGrads));
            checkCuda(cudaFree(d_oBatchMean));
            checkCuda(cudaFree(d_oBatchInvVariance));
            checkCudnn(cudnnDestroyTensorDescriptor(cudnnFullTensorDesc));
            checkCudnn(cudnnDestroyTensorDescriptor(cudnnChannelFlatTensorDesc));
        }
    }

};
