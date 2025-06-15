#pragma once
#include <exception>
#include <algorithm>
#include <cmath>
#include <cudnn_v9.h> 
#include "Layer.cuh"
#include "../kernels/Initialization.cuh"
#include "../kernels/Convolution.cuh"
#include "../network/Optimizer.cuh"
#include "../utils/Exceptions.cuh"
#include "../utils/Math.cuh"

constexpr uint BLOCK_SIZE = 256;
constexpr uint BLOCK_X_SIZE = 16;
constexpr uint BLOCK_Y_SIZE = 16;



class ConvolutionLayer : public LearnableLayer {
private:
    bool skipInputGrad;
    uint filterSize;
    uint filtersFullSize;
    float* d_oOutTensor;
    float* d_oFiltersTensor;
    float* d_oFiltersGradTensor;
    float* d_bPrevTensor;
    float* d_oCudnnWorkspace;
    size_t cudnnWorkspaceActualSizeBytes;
    cudnnHandle_t cudnnHandle;
    size_t cudnnWorkspaceFwdSizeBytes;
    size_t cudnnWorkspaceBwdDataSizeBytes;
    size_t cudnnWorkspaceBwdFilterSizeBytes;
    cudnnConvolutionFwdAlgo_t cudnnFwdAlgo;
    cudnnConvolutionBwdDataAlgo_t cudnnBwdDataAlgo;
    cudnnConvolutionBwdFilterAlgo_t cudnnBwdFilterAlgo;
    cudnnTensorDescriptor_t cudnnInTensorDesc;
    cudnnTensorDescriptor_t cudnnOutTensorDesc;
    cudnnFilterDescriptor_t cudnnFilterDesc;
    cudnnConvolutionDescriptor_t cudnnConvDesc;

public:
    ConvolutionLayer(
            const TensorSize inputSize,
            const uint outChannels,
            const uint filterSize,
            const uint stride,
            const bool skipInputGrad,
            const Optimizer::OptimAlgo algorithm,
            const cudnnHandle_t handle):
                LearnableLayer(inputSize, {}, algorithm),
                skipInputGrad(skipInputGrad),
                filterSize(filterSize),
                d_oOutTensor(nullptr),
                d_oFiltersTensor(nullptr),
                d_oFiltersGradTensor(nullptr),
                d_bPrevTensor(nullptr),
                d_oCudnnWorkspace(nullptr),
                cudnnWorkspaceActualSizeBytes(0),
                cudnnHandle(handle),
                cudnnWorkspaceFwdSizeBytes(0),
                cudnnWorkspaceBwdDataSizeBytes(0),
                cudnnWorkspaceBwdFilterSizeBytes(0),
                cudnnFwdAlgo(CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM),
                cudnnBwdDataAlgo(CUDNN_CONVOLUTION_BWD_DATA_ALGO_1),
                cudnnBwdFilterAlgo(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1) {
        // Correctly calculate the output size from input size and stride
        outputSize = {outChannels, ceilDiv(inputSize.H, stride), ceilDiv(inputSize.W, stride)};
        filtersFullSize = outputSize.C * inputSize.C * filterSize * filterSize;
        // Create the cudnn tensor, filter and convolution descriptors
        checkCudnn(cudnnCreateTensorDescriptor(&cudnnInTensorDesc));
        checkCudnn(cudnnCreateTensorDescriptor(&cudnnOutTensorDesc));
        checkCudnn(cudnnCreateFilterDescriptor(&cudnnFilterDesc));
        checkCudnn(cudnnCreateConvolutionDescriptor(&cudnnConvDesc));
        // Setup the filter and convolution cudnn descriptors 
        checkCudnn(cudnnSetFilter4dDescriptor(
            cudnnFilterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
            outChannels, inputSize.C, filterSize, filterSize
        ));
        checkCudnn(cudnnSetConvolution2dDescriptor(cudnnConvDesc,
            (filterSize - 1) / 2, (filterSize - 1) / 2, stride, stride, 1, 1,
            CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT
        ));
        // Allocate the filter weight tesnor used by both training and inference
        checkCuda(cudaMalloc(&d_oFiltersTensor, filtersFullSize * sizeof(float)));
    }


    void initWeights(const uint seed) override {
        const uint filterInSize = inputSize.C * filterSize * filterSize;
        // He-Normal initialization for the convolution filter weights
        const float range = std::sqrt(2.0f / filterInSize);
        initRandomValues<true><<<ceilDiv(filtersFullSize, BLOCK_SIZE), BLOCK_SIZE>>>(
            d_oFiltersTensor, seed, 0, range, filtersFullSize
        );
        checkCudaLastError();
    }


    void registerWeights(Optimizer& optimizer) override {
        if (!backOn) return;
        optimizer.registerLayer(
            optimizerAlgorithm,
            d_oFiltersTensor,
            d_oFiltersGradTensor,
            filtersFullSize
        );
    }


    void reRegisterGrads(Optimizer& optimizer) override {
        if (!backOn) return;
        optimizer.reRegisterLayerGrads(
            optimizerAlgorithm,
            d_oFiltersTensor,
            d_oFiltersGradTensor
        );
    }


    void toggleTrain(const bool trainOn) override {
        if (backOn == trainOn) return;
        backOn = trainOn;
        if (trainOn) {
            // Allocate resources independant on batch size
            checkCuda(cudaMalloc(&d_oFiltersGradTensor, filtersFullSize * sizeof(float)));
        }
        else {
            // Free the buffers only needed during training
            checkCuda(cudaFree(d_oFiltersGradTensor));
            d_oFiltersGradTensor = nullptr;
        }
        // Reset the batch sizes and workspace size
        //  to force buffer reallocations in forward pass
        currBatchSize = 0;
        currActualBatchSize = 0;
        cudnnWorkspaceActualSizeBytes = 0;
    }


    void forward(float* d_inputTensor, const uint batchSize) override {
        // Save the borrowed input tensor for backward pass
        d_bPrevTensor = d_inputTensor;
        // Conditionally change the tensor sizes, convolutional algorithms 
        //  and reallocate buffers on batch size change
        if (currBatchSize != batchSize) {
            currBatchSize = batchSize;
            // Change the tensor descriptions
            checkCudnn(cudnnSetTensor4dDescriptor(
                cudnnInTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                batchSize, inputSize.C, inputSize.H, inputSize.W
            ));
            checkCudnn(cudnnSetTensor4dDescriptor(
                cudnnOutTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                batchSize, outputSize.C, outputSize.H, outputSize.W
            ));
            // Reallocate buffers and rebenchmark algorithms only if the actual size is smaller
            if (currActualBatchSize < batchSize) {
                currActualBatchSize = batchSize;
                // Reallocate the output tensor buffer
                const uint fullOutSizeBytes = batchSize * outputSize.fullSize() * sizeof(float);
                checkCuda(cudaFree(d_oOutTensor));
                checkCuda(cudaMalloc(&d_oOutTensor, fullOutSizeBytes));
                // Rebenchmark the convolutional algorithms and set workspace sizes
                int returnedAlgoCount;
                // Benchmark the algorithm for the forward pass
                cudnnConvolutionFwdAlgoPerf_t fwdPerfResult;
                checkCudnn(cudnnFindConvolutionForwardAlgorithm(
                    cudnnHandle, cudnnInTensorDesc, cudnnFilterDesc, cudnnConvDesc, cudnnOutTensorDesc,
                    1, &returnedAlgoCount, &fwdPerfResult
                ));
                if (!returnedAlgoCount) {
                    throw CudnnException("Could not find a forward conv algorithm");
                }
                // Set the algorithm and workspace size for forward pass
                cudnnFwdAlgo = fwdPerfResult.algo;
                cudnnWorkspaceFwdSizeBytes = fwdPerfResult.memory;
                // Conditionally banchmark the backward algorithms and set workspace sizes
                if (backOn) {
                    if (!skipInputGrad) {
                        // Conditionally benchmark the algorithm for the backward pass for input grad
                        cudnnConvolutionBwdDataAlgoPerf_t bwdDataPerfResult;
                        checkCudnn(cudnnFindConvolutionBackwardDataAlgorithm(
                            cudnnHandle, cudnnFilterDesc, cudnnOutTensorDesc, cudnnConvDesc, cudnnInTensorDesc,
                            1, &returnedAlgoCount, &bwdDataPerfResult
                        ));
                        if (!returnedAlgoCount) {
                            throw CudnnException("Could not find a backward data conv algorithm");
                        }
                        // Set the algorithm and workspace size for backward pass for input grad
                        cudnnBwdDataAlgo = bwdDataPerfResult.algo;
                        cudnnWorkspaceBwdDataSizeBytes = bwdDataPerfResult.memory;
                    }
                    else {
                        cudnnWorkspaceBwdDataSizeBytes = 0;
                    }
                    // Benchmark the algorithm for the backward pass for filter weights grad
                    cudnnConvolutionBwdFilterAlgoPerf_t bwdFilterPerfResult;
                    checkCudnn(cudnnFindConvolutionBackwardFilterAlgorithm(
                        cudnnHandle, cudnnInTensorDesc, cudnnOutTensorDesc, cudnnConvDesc, cudnnFilterDesc,
                        1, &returnedAlgoCount, &bwdFilterPerfResult
                    ));
                    if (!returnedAlgoCount) {
                        throw CudnnException("Could not find a backward filter conv algorithm");
                    }
                    // Set the algorithm and workspace size for backward pass for filter weights grad
                    cudnnBwdFilterAlgo = bwdFilterPerfResult.algo;
                    cudnnWorkspaceBwdFilterSizeBytes = bwdFilterPerfResult.memory;
                }
                else {
                    cudnnWorkspaceBwdDataSizeBytes = 0;
                    cudnnWorkspaceBwdFilterSizeBytes = 0;
                }
                // Find maximal needed workspace size and reallocate if necessary
                uint maxWorkspaceSize = std::max({
                    cudnnWorkspaceFwdSizeBytes,
                    cudnnWorkspaceBwdDataSizeBytes,
                    cudnnWorkspaceBwdFilterSizeBytes
                });
                if (cudnnWorkspaceActualSizeBytes < maxWorkspaceSize) {
                    cudnnWorkspaceActualSizeBytes = maxWorkspaceSize;
                    checkCuda(cudaFree(d_oCudnnWorkspace));
                    checkCuda(cudaMalloc(&d_oCudnnWorkspace, cudnnWorkspaceActualSizeBytes));
                }
            }
            else {
                // Set workspace size for the forward pass with the existing algorithm
                checkCudnn(cudnnGetConvolutionForwardWorkspaceSize(
                    cudnnHandle, cudnnInTensorDesc, cudnnFilterDesc, cudnnConvDesc, cudnnOutTensorDesc,
                    cudnnFwdAlgo, &cudnnWorkspaceFwdSizeBytes
                ));
                // Conditionally set workspace sizes for the backward passes
                if (backOn) {
                    // Set workspace size for the backward data pass
                    if (!skipInputGrad) {
                        checkCudnn(cudnnGetConvolutionBackwardDataWorkspaceSize(
                            cudnnHandle, cudnnFilterDesc, cudnnOutTensorDesc, cudnnConvDesc, cudnnInTensorDesc,
                            cudnnBwdDataAlgo, &cudnnWorkspaceBwdDataSizeBytes
                        ));
                    }
                    else {
                        cudnnWorkspaceBwdDataSizeBytes = 0;
                    }
                    // Set workspace size for the backward filter pass
                    checkCudnn(cudnnGetConvolutionBackwardFilterWorkspaceSize(
                        cudnnHandle, cudnnInTensorDesc, cudnnOutTensorDesc, cudnnConvDesc, cudnnFilterDesc,
                        cudnnBwdFilterAlgo, &cudnnWorkspaceBwdFilterSizeBytes
                    ));
                }
                else {
                    cudnnWorkspaceBwdDataSizeBytes = 0;
                    cudnnWorkspaceBwdFilterSizeBytes = 0;
                }
            }
        }
        const float alpha = 1.0f;
        const float beta = 0.0f;
        // Compute the convolution forward pass to the owned output tensor
        checkCudnn(cudnnConvolutionForward(
            cudnnHandle, &alpha, cudnnInTensorDesc, d_inputTensor, cudnnFilterDesc, d_oFiltersTensor,
            cudnnConvDesc, cudnnFwdAlgo, d_oCudnnWorkspace, cudnnWorkspaceFwdSizeBytes,
            &beta, cudnnOutTensorDesc, d_oOutTensor
        ));
        // Lend the output tensor with activation data to the next layer
        next->forward(d_oOutTensor, batchSize);
    }
    
    
    void backward(float* d_gradientTensor) override {
        // Reclaim the incoming gradient tensor as owned output tensor
        d_oOutTensor = d_gradientTensor;
        // Accumulate the filter weight gradients into the owned grad tensor
        const float alpha = 1.0f;
        const float beta_accumulate = 1.0f;
        checkCudnn(cudnnConvolutionBackwardFilter(
            cudnnHandle, &alpha, cudnnInTensorDesc, d_bPrevTensor, cudnnOutTensorDesc, d_gradientTensor,
            cudnnConvDesc, cudnnBwdFilterAlgo, d_oCudnnWorkspace, cudnnWorkspaceBwdFilterSizeBytes,
            &beta_accumulate, cudnnFilterDesc, d_oFiltersGradTensor
        ));
        // Conditionally compute the input gradients into the borrowed input tensor
        const float beta_overwrite = 0.0f;
        if (!skipInputGrad) {
            checkCudnn(cudnnConvolutionBackwardData(
                cudnnHandle, &alpha, cudnnFilterDesc, d_oFiltersTensor, cudnnOutTensorDesc, d_gradientTensor,
                cudnnConvDesc, cudnnBwdDataAlgo, d_oCudnnWorkspace, cudnnWorkspaceBwdDataSizeBytes,
                &beta_overwrite, cudnnInTensorDesc, d_bPrevTensor
            ));
        }
        // Complete memory the trade by returning the input tensor
        //  conditionally with gradient data
        prev->backward(d_bPrevTensor);
    }


    ~ConvolutionLayer() override {
        if (!std::uncaught_exceptions()) {
            checkCuda(cudaFree(d_oOutTensor));
            checkCuda(cudaFree(d_oFiltersTensor));
            checkCuda(cudaFree(d_oFiltersGradTensor));
            checkCuda(cudaFree(d_oCudnnWorkspace));
            checkCudnn(cudnnDestroyTensorDescriptor(cudnnInTensorDesc));
            checkCudnn(cudnnDestroyTensorDescriptor(cudnnOutTensorDesc));
            checkCudnn(cudnnDestroyFilterDescriptor(cudnnFilterDesc));
            checkCudnn(cudnnDestroyConvolutionDescriptor(cudnnConvDesc));
        }
    }

};



template <int FILTER_R_SIZE, int STRIDE>
class DepthwiseConvolutionLayer : public LearnableLayer {
private:
    bool skipInputGrad;
    uint filtersFullSize;
    float* d_oOutTensor;
    float* d_oFiltersTensor;
    float* d_oFiltersGradTensor;
    float* d_bPrevTensor;

public:
    DepthwiseConvolutionLayer(
            const TensorSize inputSize,
            const bool skipInputGrad,
            const Optimizer::OptimAlgo algorithm):
                LearnableLayer(inputSize, {}, algorithm),
                skipInputGrad(skipInputGrad),
                filtersFullSize(inputSize.C * FILTER_R_SIZE * FILTER_R_SIZE),
                d_oOutTensor(nullptr),
                d_oFiltersTensor(nullptr),
                d_oFiltersGradTensor(nullptr),
                d_bPrevTensor(nullptr) {
        // Correctly calculate the output size from input size and stride
        outputSize = {inputSize.C, ceilDiv(inputSize.H, STRIDE), ceilDiv(inputSize.W, STRIDE)};
        // Allocate the output tensor use by both training and inference
        checkCuda(cudaMalloc(&d_oFiltersTensor, filtersFullSize * sizeof(float)));
    }


    void initWeights(const uint seed) override {
        // He-Normal initialization for the convolution filter weights
        const float range = std::sqrt(2.0f / (FILTER_R_SIZE * FILTER_R_SIZE));
        initRandomValues<true><<<ceilDiv(filtersFullSize, BLOCK_SIZE), BLOCK_SIZE>>>(
            d_oFiltersTensor, seed, 0, range, filtersFullSize
        );
        checkCudaLastError();
    }


    void registerWeights(Optimizer& optimizer) override {
        if (!backOn) return;
        optimizer.registerLayer(
            optimizerAlgorithm,
            d_oFiltersTensor,
            d_oFiltersGradTensor,
            filtersFullSize
        );
    }


    void reRegisterGrads(Optimizer& optimizer) override {
        if (!backOn) return;
        optimizer.reRegisterLayerGrads(
            optimizerAlgorithm,
            d_oFiltersTensor,
            d_oFiltersGradTensor
        );
    }


    void toggleTrain(const bool trainOn) override {
        if (backOn == trainOn) return;
        backOn = trainOn;
        if (trainOn) {
            // Allocate resources independant on batch size
            checkCuda(cudaMalloc(&d_oFiltersGradTensor, filtersFullSize * sizeof(float)));
        }
        else {
            // Free the buffers only needed during training
            checkCuda(cudaFree(d_oFiltersGradTensor));
            d_oFiltersGradTensor = nullptr;
        }
        // Reset the batch sizes to force buffer reallocations in forward pass
        currBatchSize = 0;
        currActualBatchSize = 0;
    }


    void forward(float* d_inputTensor, const uint batchSize) override {
        // Save the borrowed input tensor and batch size for backward pass
        d_bPrevTensor = d_inputTensor;
        currBatchSize = batchSize;
        // Reallocate the owned output tensor only if the actual size is smaller
        if (currActualBatchSize < batchSize) {
            currActualBatchSize = batchSize;
            checkCuda(cudaFree(d_oOutTensor));
            const uint fullOutSize = batchSize * outputSize.fullSize();
            checkCuda(cudaMalloc(&d_oOutTensor, fullOutSize * sizeof(float)));
        }
        // Compute the convolution forward pass to the owned output tensor
        const dim3 blockSize(BLOCK_X_SIZE, BLOCK_Y_SIZE);
        const uint outHBlocks = ceilDiv(outputSize.H, BLOCK_Y_SIZE);
        const dim3 gridSize(ceilDiv(outputSize.W, BLOCK_X_SIZE), batchSize * inputSize.C * outHBlocks);
        depthwiseConvForward<BLOCK_X_SIZE, BLOCK_Y_SIZE, FILTER_R_SIZE, STRIDE><<<gridSize, blockSize>>>(
            d_oOutTensor, d_inputTensor, d_oFiltersTensor,
            inputSize.C, outputSize.H, outputSize.W, outHBlocks,
            inputSize.H, inputSize.W
        );
        checkCudaLastError();
        // Lend the output tensor with activation data to the next layer
        next->forward(d_oOutTensor, batchSize);
    }
    

    void backward(float* d_gradientTensor) override {
        // Reclaim the incoming gradient tensor as owned output tensor
        d_oOutTensor = d_gradientTensor;
        // Accumulate the filter weight gradients into the owned grad tensor
        const uint gridSize = inputSize.C * FILTER_R_SIZE * FILTER_R_SIZE;
        depthwiseConvBackwardGrad<BLOCK_SIZE, FILTER_R_SIZE, STRIDE><<<gridSize, BLOCK_SIZE>>>(
            d_oFiltersGradTensor, d_gradientTensor, d_bPrevTensor, currBatchSize, inputSize.C,
            outputSize.H * outputSize.W, outputSize.W, inputSize.H, inputSize.W
        );
        checkCudaLastError();
        // Conditionally compute the input gradients into the borrowed input tensor
        if (!skipInputGrad) {
            const dim3 blockSize(BLOCK_X_SIZE, BLOCK_Y_SIZE);
            const uint inHBlocks = ceilDiv(inputSize.H, BLOCK_Y_SIZE);
            const dim3 gridSize(ceilDiv(inputSize.W, BLOCK_X_SIZE), currBatchSize * inputSize.C * inHBlocks);
            depthwiseConvBackward<BLOCK_X_SIZE, BLOCK_Y_SIZE, FILTER_R_SIZE, STRIDE><<<gridSize, blockSize>>>(
                d_bPrevTensor, d_gradientTensor, d_oFiltersTensor,
                inputSize.C, inputSize.H, inputSize.W, inHBlocks,
                outputSize.H, outputSize.W
            );
            checkCudaLastError();
        }
        // Complete memory the trade by returning the input tensor
        //  conditionally with gradient data
        prev->backward(d_bPrevTensor);
    }


    ~DepthwiseConvolutionLayer() override {
        if (!std::uncaught_exceptions()) {
            checkCuda(cudaFree(d_oOutTensor));
            checkCuda(cudaFree(d_oFiltersTensor));
            checkCuda(cudaFree(d_oFiltersGradTensor));
        }
    }

};
