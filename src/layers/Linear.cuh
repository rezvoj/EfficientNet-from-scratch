#pragma once
#include <exception>
#include <cublas_v2.h> 
#include "Layer.cuh"
#include "../kernels/Initialization.cuh"
#include "../kernels/Resizing.cuh"
#include "../network/Optimizer.cuh"
#include "../utils/Exceptions.cuh"
#include "../utils/Math.cuh"

constexpr size_t BLOCK_SIZE = 256;



class LinearLayer : public LearnableLayer {
private:
    bool skipInputGrad;
    size_t weightsFullSize;
    float* d_oOutMatrix;
    float* d_oWeightMatrix;
    float* d_oWeightGradMatrix;
    float* d_oBiasVector;
    float* d_oBiasGradVector;
    float* d_oBatchOnesVector;
    float* d_bPrevMatrix;
    cublasHandle_t cublasHandle;

public:
    LinearLayer(
            const size_t inNeurons,
            const size_t outNeurons,
            const bool skipInputGrad,
            const Optimizer::OptimAlgo algorithm,
            const cublasHandle_t handle):
                LearnableLayer({inNeurons, 1, 1}, {outNeurons, 1, 1}, algorithm),
                skipInputGrad(skipInputGrad),
                weightsFullSize(outNeurons * inNeurons),
                d_oOutMatrix(nullptr),
                d_oWeightMatrix(nullptr),
                d_oWeightGradMatrix(nullptr),
                d_oBiasVector(nullptr),
                d_oBiasGradVector(nullptr),
                d_oBatchOnesVector(nullptr),
                d_bPrevMatrix(nullptr),
                cublasHandle(handle) {
        checkCuda(cudaMalloc(&d_oWeightMatrix, weightsFullSize * sizeof(float)));
        checkCuda(cudaMalloc(&d_oBiasVector, outputSize.C * sizeof(float)));
    }


    void initWeights(const size_t seed) override {
        // He-Normal initialization for the linear layer weights
        const float range = std::sqrt(2.0f / inputSize.C);
        initRandomValues<true><<<ceilDiv(weightsFullSize, BLOCK_SIZE), BLOCK_SIZE>>>(
            d_oWeightMatrix, seed, 0, range, weightsFullSize
        );
        checkCudaLastError();
        // Zero initialize the biases
        initValues<<<ceilDiv(outputSize.C, BLOCK_SIZE), BLOCK_SIZE>>>(
            d_oBiasVector, 0.0f, outputSize.C
        );
        checkCudaLastError();
    }


    void registerWeights(Optimizer& optimizer) override {
        if (!backOn) return;
        optimizer.registerLayer(optimizerAlgorithm, d_oWeightMatrix, d_oWeightGradMatrix, weightsFullSize);
        optimizer.registerLayer(optimizerAlgorithm, d_oBiasVector, d_oBiasGradVector, outputSize.C);
    }


    void reRegisterGrads(Optimizer& optimizer) override {
        if (!backOn) return;
        optimizer.reRegisterLayerGrads(optimizerAlgorithm, d_oWeightMatrix, d_oWeightGradMatrix);
        optimizer.reRegisterLayerGrads(optimizerAlgorithm, d_oBiasVector, d_oBiasGradVector);
    }


    void toggleTrain(const bool trainOn) override {
        if (backOn == trainOn) return;
        backOn = trainOn;
        if (trainOn) {
            // Allocate resources independant on batch size
            checkCuda(cudaMalloc(&d_oWeightGradMatrix, weightsFullSize * sizeof(float)));
            checkCuda(cudaMalloc(&d_oBiasGradVector, outputSize.C * sizeof(float)));
        }
        else {
            // Free the unnecessary memory for inference
            checkCuda(cudaFree(d_oWeightGradMatrix));
            checkCuda(cudaFree(d_oBiasGradVector));
            checkCuda(cudaFree(d_oBatchOnesVector));
            d_oWeightGradMatrix = nullptr;
            d_oBiasGradVector = nullptr;
            d_oBatchOnesVector = nullptr;
        }
        // Reset the batch sizes to force buffer reallocations
        currBatchSize = 0;
        currActualBatchSize = 0;
    }


    void forward(float* d_inputMatrix, const size_t batchSize) override {
        // Save the borrowed input matrix and batch size for backward pass
        d_bPrevMatrix = d_inputMatrix;
        currBatchSize = batchSize;
        // Conditionally reallocate the output matrix if buffer is too small
        if (currActualBatchSize < batchSize) {
            currActualBatchSize = batchSize;
            const size_t fullOutSizeBytes = batchSize * outputSize.fullSize() * sizeof(float);
            checkCuda(cudaFree(d_oOutMatrix));
            checkCuda(cudaMalloc(&d_oOutMatrix, fullOutSizeBytes));
            if (backOn) {
                // Reinitialize the vector of ones with the size of a batch
                checkCuda(cudaFree(d_oBatchOnesVector));
                checkCuda(cudaMalloc(&d_oBatchOnesVector, batchSize * sizeof(float)));
                initValues<<<ceilDiv(batchSize, BLOCK_SIZE), BLOCK_SIZE>>>(
                    d_oBatchOnesVector, 1.0f, batchSize
                );
                checkCudaLastError();
            }
        }
        // Computes the activations into the owned output matrix
        // The matmul operation from the row-wise perspective: 
        //  (N, C_in) @ (C_out, C_in)^T = (N, C_out)
        const float alpha = 1.0f;
        const float beta = 0.0f;
        checkCublas(cublasSgemm_v2(
            cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
            outputSize.C, batchSize, inputSize.C,
            &alpha, d_oWeightMatrix, inputSize.C, 
            d_inputMatrix, inputSize.C,
            &beta, d_oOutMatrix, outputSize.C
        )); 
        // Add the bias vector to each row of the output matrix
        const size_t fullOutSize = currBatchSize * outputSize.C;
        broadcastAddBiasInplace<<<ceilDiv(fullOutSize, BLOCK_SIZE), BLOCK_SIZE>>>(
            d_oOutMatrix, d_oBiasVector, outputSize.C, fullOutSize
        );
        checkCudaLastError();
        // Lend the output matrix with activation data to the next layer
        next->forward(d_oOutMatrix, batchSize);
    }


    void backward(float* d_gradientMatrix) override {
        // Reclaim the incoming gradient matrix as owned output matrix
        d_oOutMatrix = d_gradientMatrix;
        // Accumulate the bias gradients into the owned bias grad vector
        // The matmul operation from the row-wise perspective: 
        //  (N) @ (N, C_out) = (C_out)
        const float alpha = 1.0f;
        const float beta_accumulate = 1.0f;
        checkCublas(cublasSgemv_v2(
            cublasHandle, CUBLAS_OP_N, outputSize.C, currBatchSize,
            &alpha, d_gradientMatrix, outputSize.C,
            d_oBatchOnesVector, 1,
            &beta_accumulate, d_oBiasGradVector, 1
        ));
        // Accumulate the weight gradients into the owned weight grad matrix
        // The matmul operation from the row-wise perspective: 
        //  (N, C_out)^T @ (N, C_in) = (C_out, C_in)
        checkCublas(cublasSgemm_v2(
            cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
            inputSize.C, outputSize.C, currBatchSize,
            &alpha, d_bPrevMatrix, inputSize.C,
            d_gradientMatrix, outputSize.C,
            &beta_accumulate, d_oWeightGradMatrix, inputSize.C
        ));
        // Conditionally compute the input matrix gradient into borrowed prev matrix
        if (!skipInputGrad) {
            // The matmul operation from the row-wise perspective:
            //  (N, C_out) @ (C_out, C_in) = (N, C_in)
            const float beta_overwrite = 0.0f;
            checkCublas(cublasSgemm_v2(
                cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                inputSize.C, currBatchSize, outputSize.C,
                &alpha, d_oWeightMatrix, inputSize.C,
                d_gradientMatrix, outputSize.C,
                &beta_overwrite, d_bPrevMatrix, inputSize.C
            ));
        }
        // Complete memory the trade by returning the input matrix
        //  conditionally with gradient data
        prev->backward(d_bPrevMatrix);
    }


    ~LinearLayer() override {
        if (!std::uncaught_exceptions()) {
            checkCuda(cudaFree(d_oOutMatrix));
            checkCuda(cudaFree(d_oWeightMatrix));
            checkCuda(cudaFree(d_oWeightGradMatrix));
            checkCuda(cudaFree(d_oBiasVector));
            checkCuda(cudaFree(d_oBiasGradVector));
            checkCuda(cudaFree(d_oBatchOnesVector));
        }
    }

};
