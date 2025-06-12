#include <stdexcept>
#include <random>
#include <cudnn_v9.h>
#include "Kernels.cuh"
#include "Utils.cuh"

constexpr size_t BLOCK_SIZE = 256;
constexpr size_t BLOCK_X_SIZE = 16;
constexpr size_t BLOCK_Y_SIZE = 16;



struct TensorSize {
    size_t C = 1;
    size_t H = 1;
    size_t W = 1;
    
    size_t fullSize() const {
        return C * H * W;
    }

    bool operator==(const TensorSize& other) const {
        return C == other.C
                && H == other.H
                && W == other.W;
    }

    bool operator!=(const TensorSize& other) const {
        return !(*this == other);
    }
};



class Optimizer {



};



// When grad is OFF during inference the layer just borrows the memory to the next layer.
// During training the model expects to have the memory it borrowed or same size memory
// to get returned (traded back) by the following layer fulfilling the "Memory Contract".
// After forward call when grad is ON, the layer is in non atomic and destroying it or
// toggling grad would result in undefined state or crash.
class Layer {
protected:
    TensorSize inputSize;
    TensorSize outputSize;
    size_t currBatchSize;
    Layer *prev, *next;

public:
    Layer(const TensorSize inputSize, const TensorSize outputSize):
            inputSize(inputSize),
            outputSize(outputSize),
            currBatchSize(0),
            prev(nullptr),
            next(nullptr) {}
    virtual ~Layer() {}
    void setPrev(Layer* prevLayer) { prev = prevLayer; }
    virtual void append(Layer* nextLayer) {
        next = nextLayer;
        nextLayer->setPrev(this);
    }
    virtual void toggleGrad([[maybe_unused]] const bool gradON) {};
    virtual void forward(float* d_inputTensor, const size_t batchSize) = 0;
    virtual void backward(float* d_gradientTensor) = 0;
    virtual void optimize(const Optimizer& optimizer) { next->optimize(optimizer); }
};



// The layers must be appended in the same order as the corresponding 
// merge layer is later connected to the previous layers.
class SplitLayer : public Layer {
private:
    float* d_copyTensor;
    Layer* next2;
    bool partialBackward;

public:
    SplitLayer(const TensorSize inputSize): 
            Layer(inputSize, inputSize),
            d_copyTensor(nullptr),
            next2(nullptr),
            partialBackward(false) {}

    void append(Layer* nextLayer) override {
        if (next == nullptr) next = nextLayer;
        else next2 = nextLayer;
        nextLayer->setPrev(this);
    }

    void forward(float* d_borrowTensor, const size_t batchSize) override {
        const size_t fullSizeBytes = batchSize * inputSize.fullSize() * sizeof(float);
        if (currBatchSize != batchSize) {
            currBatchSize = batchSize;
            checkCuda(cudaFree(d_copyTensor));
            checkCuda(cudaMalloc(&d_copyTensor, fullSizeBytes));
        }
        checkCuda(cudaMemcpy(d_copyTensor, d_borrowTensor, fullSizeBytes, cudaMemcpyDeviceToDevice));
        next->forward(d_borrowTensor, batchSize);
        next2->forward(d_copyTensor, batchSize);
    }

    void backward(float* d_replaceTensor) override {
        if (!partialBackward) {
            d_copyTensor = d_replaceTensor;
            partialBackward = true;
            return;
        }
        partialBackward = false;
        const size_t fullSize = currBatchSize * inputSize.fullSize();
        elementwiseAddInplace<<<ceilDiv(fullSize, BLOCK_SIZE), BLOCK_SIZE>>>(
            d_replaceTensor, d_copyTensor, fullSize
        );
        checkCudaLastError();
        prev->backward(d_replaceTensor);
    }

    void optimize(const Optimizer& optimizer) override {
        next->optimize(optimizer);
        next2->optimize(optimizer);
    }

    ~SplitLayer() override {
        checkCuda(cudaFree(d_copyTensor));
    }
};



// The layer must be appended to the previous layers in the same order as 
// the paths were appended into the corresponding split layer.
class MulMergeBroadcastedLayer : public Layer {
private:
    bool backON;
    float* d_outTensor;
    Layer* prev2;
    float* d_prev1Tensor;
    float* d_prev2Tensor;
    bool partialForward;

public:
    MulMergeBroadcastedLayer(const TensorSize inputSize): 
            Layer(inputSize, inputSize),
            d_outTensor(nullptr),
            prev2(nullptr),
            d_prev1Tensor(nullptr),
            d_prev2Tensor(nullptr),
            partialForward(false) {}

     void toggleGrad(const bool gradON) override {
        if (!backON && gradON) backON = true;
        else if (backON && !gradON) {
            backON = false;
            checkCuda(cudaFree(d_outTensor));
            d_outTensor = nullptr;
            currBatchSize = 0;
        }    
    };

    void forward(float* d_borrowTensor, const size_t batchSize) override {
        const size_t fullSize = batchSize * inputSize.fullSize();
        if (!partialForward) {
            if (backON && currBatchSize != batchSize) {
                currBatchSize = batchSize;
                checkCuda(cudaFree(d_outTensor));
                checkCuda(cudaMalloc(&d_outTensor, fullSize * sizeof(float)));
            }
            d_prev1Tensor = d_borrowTensor;
            return;
        }
        if (backON) {
            d_prev2Tensor = d_borrowTensor;
            elementwiseMul<<<ceilDiv(fullSize, BLOCK_SIZE), BLOCK_SIZE>>>(
                d_outTensor, d_prev1Tensor, d_prev2Tensor, fullSize
            );
            checkCudaLastError();
            next->forward(d_outTensor, batchSize);
            return;
        }
        elementwiseMulInplace<<<ceilDiv(fullSize, BLOCK_SIZE), BLOCK_SIZE>>>(
            d_prev1Tensor, d_prev2Tensor, fullSize
        );
        checkCudaLastError();
        next->forward(d_prev1Tensor, batchSize);
    }

    void backward(float* d_replaceTensor) override {
        d_outTensor = d_replaceTensor;
        const size_t fullSize = currBatchSize * inputSize.fullSize();
        elementwiseMulBackwardInplace<<<ceilDiv(fullSize, BLOCK_SIZE), BLOCK_SIZE>>>(
            d_prev2Tensor, d_prev1Tensor, d_outTensor, fullSize
        ); 
        checkCudaLastError();
        prev2->backward(d_prev2Tensor);
        prev->backward(d_prev1Tensor);
        d_prev2Tensor = nullptr;
        d_prev1Tensor = nullptr;
    }

    ~MulMergeBroadcastedLayer() override {
        checkCuda(cudaFree(d_outTensor));
    }
};



// The layer must be appended to the previous layers in the same order as 
// the paths were appended into the corresponding split layer.
class AddMergeLayer : public Layer {
private:
    Layer* prev2;
    float* d_prevTensor;
    bool partialForward;

public:
    AddMergeLayer(const TensorSize inputSize): 
            Layer(inputSize, inputSize),
            prev2(nullptr),
            d_prevTensor(nullptr),
            partialForward(false) {}

    void forward(float* d_borrowTensor, const size_t batchSize) override {
        if (!partialForward) {
            currBatchSize = batchSize;
            d_prevTensor = d_borrowTensor;
            return;
        }
        const size_t fullSize = batchSize * inputSize.fullSize();
        elementwiseAddInplace<<<ceilDiv(fullSize, BLOCK_SIZE), BLOCK_SIZE>>>(
            d_borrowTensor, d_prevTensor, fullSize
        ); checkCudaLastError();
        next->forward(d_borrowTensor, batchSize);
    }

    void backward(float* d_replaceTensor) override {
        const size_t copySize = currBatchSize * inputSize.fullSize() * sizeof(float);        
        checkCuda(cudaMemcpy(d_prevTensor, d_replaceTensor, copySize, cudaMemcpyDeviceToDevice));
        prev2->backward(d_replaceTensor);
        prev->backward(d_prevTensor);
        d_prevTensor = nullptr;
    }
};



class StochasticDepthLayer : public Layer {
private:
    bool backON;
    bool skippedForward;
    float retainRate;
    std::mt19937 randomGenerator;
    std::uniform_real_distribution<> distribution;

public:
    StochasticDepthLayer(const TensorSize inputSize, const float rate): 
            Layer(inputSize, inputSize),
            backON(false),
            skippedForward(false),
            retainRate(rate) {
        std::random_device device;
        randomGenerator = std::mt19937(device());
        distribution = std::uniform_real_distribution<>(0.0, 1.0);
    }

    void toggleGrad(const bool gradON) override { backON = gradON; };

    void forward(float* d_borrowTensor, const size_t batchSize) override {
        currBatchSize = batchSize;
        float multiplier = 1.0;
        if (!backON) multiplier = retainRate;
        else if (distribution(randomGenerator) < static_cast<double>(retainRate)) {
            multiplier = 0.0f;
            skippedForward = true;
        }
        if (multiplier != 1.0f) {
            const size_t fullSize = batchSize * inputSize.fullSize();
            elementwiseScalarMulInplace<<<ceilDiv(fullSize, BLOCK_SIZE), BLOCK_SIZE>>>(
                d_borrowTensor, retainRate, fullSize
            );
            checkCudaLastError();
        }    
        next->forward(d_borrowTensor, batchSize);
    }

    void backward(float* d_replaceTensor) override {
        if (skippedForward) {
            skippedForward = false;
            const size_t fullSize = currBatchSize * inputSize.fullSize();
            elementwiseScalarMulInplace<<<ceilDiv(fullSize, BLOCK_SIZE), BLOCK_SIZE>>>(
                d_replaceTensor, 0.0f, fullSize
            );
            checkCudaLastError();
        }
        prev->backward(d_replaceTensor);
    }
};



class DropoutLayer : public Layer {
private:
    bool backON;
    float dropoutRate;
    size_t dropoutSeed;
    void* d_cudnnDropoutStates;
    void* d_cudnnReserveSpace;
    cudnnHandle_t cudnnHandle;
    size_t cudnnDropoutStatesSize;
    size_t cudnnReserveSpaceSize;
    cudnnDropoutDescriptor_t cudnnDropoutDesc;
    cudnnTensorDescriptor_t cudnnTensorDesc;

public:
    DropoutLayer(
            const TensorSize inputSize,
            const float rate,
            const size_t seed,
            const cudnnHandle_t handle): 
                Layer(inputSize, inputSize),
                backON(false),
                dropoutRate(rate),
                dropoutSeed(seed),
                d_cudnnDropoutStates(nullptr),
                d_cudnnReserveSpace(nullptr),
                cudnnHandle(handle),
                cudnnReserveSpaceSize(0) {
        checkCudnn(cudnnCreateDropoutDescriptor(&cudnnDropoutDesc));
        checkCudnn(cudnnDropoutGetStatesSize(cudnnHandle, &cudnnDropoutStatesSize));
        checkCudnn(cudnnCreateTensorDescriptor(&cudnnTensorDesc));
    }

    void toggleGrad(const bool gradON) override {
        if (!backON && gradON) {
            backON = true;
            checkCuda(cudaMalloc(&d_cudnnDropoutStates, cudnnDropoutStatesSize));
            checkCudnn(cudnnSetDropoutDescriptor(
                cudnnDropoutDesc, cudnnHandle, dropoutRate,
                d_cudnnDropoutStates, cudnnDropoutStatesSize, dropoutSeed
            ));
        }
        else if (backON && !gradON) {
            backON = false;
            checkCuda(cudaFree(d_cudnnDropoutStates));
            checkCuda(cudaFree(d_cudnnReserveSpace));
            d_cudnnDropoutStates = nullptr;
            d_cudnnReserveSpace = nullptr;
            cudnnReserveSpaceSize = 0;
            currBatchSize = 0;
        }
    };

    void forward(float* d_borrowTensor, const size_t batchSize) override {
        if (backON) {
            if (currBatchSize != batchSize) {
                currBatchSize = batchSize;
                checkCudnn(cudnnSetTensor4dDescriptor(
                    cudnnTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                    batchSize, inputSize.C, inputSize.H, inputSize.W
                ));
                checkCuda(cudaFree(d_cudnnReserveSpace));
                checkCudnn(cudnnDropoutGetReserveSpaceSize(cudnnTensorDesc, &cudnnReserveSpaceSize));
                checkCuda(cudaMalloc(&d_cudnnReserveSpace, cudnnReserveSpaceSize));
            }
            checkCudnn(cudnnDropoutForward(
                cudnnHandle, cudnnDropoutDesc,
                cudnnTensorDesc, d_borrowTensor,
                cudnnTensorDesc, d_borrowTensor,   
                d_cudnnReserveSpace, cudnnReserveSpaceSize
            ));
        }
        next->forward(d_borrowTensor, batchSize);
    }

    void backward(float* d_replaceTensor) override {
        checkCudnn(cudnnDropoutBackward(
            cudnnHandle, cudnnDropoutDesc,
            cudnnTensorDesc, d_replaceTensor,
            cudnnTensorDesc, d_replaceTensor,   
            d_cudnnReserveSpace, cudnnReserveSpaceSize
        ));
        prev->backward(d_replaceTensor);
    }

    ~DropoutLayer() override {
        checkCuda(cudaFree(d_cudnnDropoutStates));
        checkCuda(cudaFree(d_cudnnReserveSpace));
        checkCudnn(cudnnDestroyDropoutDescriptor(cudnnDropoutDesc));
        checkCudnn(cudnnDestroyTensorDescriptor(cudnnTensorDesc));
    }
};



template <typename ACTIVATION>
class ActivationLayer : public Layer {
private:
    bool backON;
    float* d_Tensor;

public:
    ActivationLayer(const TensorSize inputSize): 
            Layer(inputSize, inputSize),
            backON(false),
            d_Tensor(nullptr) {}
    
    void toggleGrad(const bool gradON) override {
        if (!backON && gradON) backON = true;
        else if (backON && !gradON) {
            backON = false;
            checkCuda(cudaFree(d_Tensor));
            d_Tensor = nullptr;
            currBatchSize = 0;
        }
    };

    void forward(float* d_borrowTensor, const size_t batchSize) override {
        const size_t fullSize = batchSize * inputSize.fullSize();
        if (backON) {
            if (currBatchSize != batchSize) {
                currBatchSize = batchSize;
                checkCuda(cudaFree(d_Tensor));
                checkCuda(cudaMalloc(&d_Tensor, fullSize * sizeof(float)));
            }
            elementwiseActivation<ACTIVATION>
                <<<ceilDiv(fullSize, BLOCK_SIZE), BLOCK_SIZE>>>(d_Tensor, d_borrowTensor, fullSize); 
            checkCudaLastError();
            std::swap(d_borrowTensor, d_Tensor);
        }
        else {
            elementwiseActivationInplace<ACTIVATION>
                <<<ceilDiv(fullSize, BLOCK_SIZE), BLOCK_SIZE>>>(d_borrowTensor, fullSize);
            checkCudaLastError();
        }
        next->forward(d_borrowTensor, batchSize);
    }

    void backward(float* d_replaceTensor) override {
        const size_t fullSize = currBatchSize * inputSize.fullSize();
        elementwiseActivationBackwardInplace<ACTIVATION>
            <<<ceilDiv(fullSize, BLOCK_SIZE), BLOCK_SIZE>>>(d_replaceTensor, d_Tensor, fullSize); 
        checkCudaLastError();
        prev->backward(d_replaceTensor);
    }

    ~ActivationLayer() override {
        checkCuda(cudaFree(d_Tensor));
    }
};



class ExpansionLayer : public Layer {
private:    
    float* d_outputTensor;
    float* d_prevTensor;
    cudnnHandle_t cudnnHandle;
    cudnnReduceTensorDescriptor_t cudnnReduceDesc;
    cudnnTensorDescriptor_t cudnnInTensorDesc;
    cudnnTensorDescriptor_t cudnnOutTensorDesc;

public:
    ExpansionLayer(const TensorSize outputSize, const cudnnHandle_t handle):
            Layer({outputSize.C, 1, 1}, outputSize),
            d_outputTensor(nullptr),
            d_prevTensor(nullptr),
            cudnnHandle(handle) {
        checkCudnn(cudnnCreateReduceTensorDescriptor(&cudnnReduceDesc));
        checkCudnn(cudnnCreateTensorDescriptor(&cudnnInTensorDesc));
        checkCudnn(cudnnCreateTensorDescriptor(&cudnnOutTensorDesc));
        checkCudnn(cudnnSetReduceTensorDescriptor(
            cudnnReduceDesc, CUDNN_REDUCE_TENSOR_ADD, CUDNN_DATA_FLOAT,
            CUDNN_NOT_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES
        ));
    }

    void forward(float* d_borrowTensor, const size_t batchSize) override {
        d_prevTensor = d_borrowTensor;
        const size_t fullSize = batchSize * outputSize.fullSize();
        if (currBatchSize != batchSize) {
            currBatchSize = batchSize;
            checkCuda(cudaFree(d_outputTensor));
            checkCuda(cudaMalloc(&d_outputTensor, fullSize * sizeof(float)));
            checkCudnn(cudnnSetTensor4dDescriptor(
                cudnnInTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                batchSize, outputSize.C, 1, 1
            ));
            checkCudnn(cudnnSetTensor4dDescriptor(
                cudnnOutTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                batchSize, outputSize.C, outputSize.H, outputSize.W
            ));
        }
        tensorExpansion<<<ceilDiv(fullSize, BLOCK_SIZE), BLOCK_SIZE>>>(
            d_outputTensor, d_prevTensor, 1.0f, fullSize, outputSize.H * outputSize.W
        );
        checkCudaLastError();
        next->forward(d_outputTensor, batchSize);
    }

    void backward(float* d_replaceTensor) override {
        d_outputTensor = d_replaceTensor;
        const float alpha = 1.0f;
        const float beta = 0.0f;
        checkCudnn(cudnnReduceTensor(
            cudnnHandle, cudnnReduceDesc,
            nullptr, 0, nullptr, 0,
            &alpha, cudnnOutTensorDesc, d_outputTensor,
            &beta, cudnnInTensorDesc, d_prevTensor
        ));
        prev->backward(d_prevTensor);
    }

    ~ExpansionLayer() override {
        checkCuda(cudaFree(d_outputTensor));
        checkCudnn(cudnnDestroyReduceTensorDescriptor(cudnnReduceDesc));
        checkCudnn(cudnnDestroyTensorDescriptor(cudnnInTensorDesc));
        checkCudnn(cudnnDestroyTensorDescriptor(cudnnOutTensorDesc));
    }
};



class GlobalAvgPoolingLayer : public Layer {
private:
    float* d_outputTensor;
    float* d_prevTensor;
    cudnnHandle_t cudnnHandle;
    cudnnPoolingDescriptor_t cudnnPoolingDesc;
    cudnnTensorDescriptor_t cudnnInTensorDesc;
    cudnnTensorDescriptor_t cudnnOutTensorDesc;

public:
    GlobalAvgPoolingLayer(const TensorSize inputSize, const cudnnHandle_t handle):
            Layer(inputSize, {inputSize.C, 1, 1}),
            d_outputTensor(nullptr),
            d_prevTensor(nullptr),
            cudnnHandle(handle) {
        checkCudnn(cudnnCreatePoolingDescriptor(&cudnnPoolingDesc));
        checkCudnn(cudnnCreateTensorDescriptor(&cudnnInTensorDesc));
        checkCudnn(cudnnCreateTensorDescriptor(&cudnnOutTensorDesc));
        checkCudnn(cudnnSetPooling2dDescriptor(
            cudnnPoolingDesc, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
            CUDNN_NOT_PROPAGATE_NAN, inputSize.H, inputSize.W, 0, 0, 1, 1
        ));
    }

    void forward(float* d_borrowTensor, const size_t batchSize) override {
        d_prevTensor = d_borrowTensor;
        if (currBatchSize != batchSize) {
            currBatchSize = batchSize;
            checkCuda(cudaFree(d_outputTensor));
            checkCuda(cudaMalloc(&d_outputTensor, batchSize * outputSize.fullSize() * sizeof(float)));
            checkCudnn(cudnnSetTensor4dDescriptor(
                cudnnInTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                batchSize, inputSize.C, inputSize.H, inputSize.W
            ));
            checkCudnn(cudnnSetTensor4dDescriptor(
                cudnnOutTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                batchSize, outputSize.C, 1, 1
            ));
        }
        const float alpha = 1.0f;
        const float beta = 0.0f;
        checkCudnn(cudnnPoolingForward(
            cudnnHandle, cudnnPoolingDesc,
            &alpha, cudnnInTensorDesc, d_prevTensor,
            &beta, cudnnOutTensorDesc, d_outputTensor
        ));
        next->forward(d_outputTensor, batchSize);
    }

    void backward(float* d_replaceTensor) override {
        d_outputTensor = d_replaceTensor;
        const size_t fullSize = currBatchSize * inputSize.fullSize();
        const float scale = 1.0f / static_cast<float>(inputSize.H * inputSize.W);
        tensorExpansion<<<ceilDiv(fullSize, BLOCK_SIZE), BLOCK_SIZE>>>(
            d_prevTensor, d_outputTensor, scale, fullSize, inputSize.H * inputSize.W
        );
        checkCudaLastError();
        prev->backward(d_prevTensor);
    }

    ~GlobalAvgPoolingLayer() {
        checkCuda(cudaFree(d_outputTensor));
        checkCudnn(cudnnDestroyPoolingDescriptor(cudnnPoolingDesc));
        checkCudnn(cudnnDestroyTensorDescriptor(cudnnInTensorDesc));
        checkCudnn(cudnnDestroyTensorDescriptor(cudnnOutTensorDesc));
    }
};



class BatchNormLayer : public Layer {
private:
    bool backON;
    float expAvgFactor;
    float epsilon;
    float* d_outputTensor;
    float* d_backOutputTensor;
    float* d_scale;
    float* d_shift;
    float* d_runningMean;
    float* d_runningVar;
    float* d_scaleGrads;
    float* d_shiftGrads;
    float* d_batchMean;
    float* d_batchInvVariance;
    float* d_prevTensor;
    cudnnHandle_t cudnnHandle;
    cudnnTensorDescriptor_t cudnnFullTensorDesc;
    cudnnTensorDescriptor_t cudnnChannelFlatTensorDesc;

public:
    BatchNormLayer(
            const TensorSize inputSize,
            const float expAvgFactor,
            const float epsilon,
            const cudnnHandle_t handle):
                Layer(inputSize, inputSize),
                backON(false),
                expAvgFactor(expAvgFactor),
                epsilon(epsilon),
                d_outputTensor(nullptr),
                d_backOutputTensor(nullptr),
                d_scale(nullptr),
                d_shift(nullptr),
                d_runningMean(nullptr),
                d_runningVar(nullptr),
                d_scaleGrads(nullptr),
                d_shiftGrads(nullptr),
                d_batchMean(nullptr),
                d_batchInvVariance(nullptr),
                d_prevTensor(nullptr),
                cudnnHandle(handle) {
        checkCudnn(cudnnCreateTensorDescriptor(&cudnnFullTensorDesc));
        checkCudnn(cudnnCreateTensorDescriptor(&cudnnChannelFlatTensorDesc));
        checkCudnn(cudnnSetTensor4dDescriptor(
            cudnnChannelFlatTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
            1, inputSize.C, 1, 1
        ));
        checkCuda(cudaMalloc(&d_scale, outputSize.C * sizeof(float)));
        checkCuda(cudaMalloc(&d_shift, outputSize.C * sizeof(float)));
        checkCuda(cudaMalloc(&d_runningMean, outputSize.C * sizeof(float)));
        checkCuda(cudaMalloc(&d_runningVar, outputSize.C * sizeof(float)));
    }

    void toggleGrad(const bool gradON) override {
        if (!backON && gradON) {
            backON = true;
            checkCuda(cudaMalloc(&d_backOutputTensor, currBatchSize * inputSize.fullSize() * sizeof(float)));
            checkCuda(cudaMalloc(&d_scaleGrads, outputSize.C * sizeof(float)));
            checkCuda(cudaMalloc(&d_shiftGrads, outputSize.C * sizeof(float)));
            checkCuda(cudaMalloc(&d_batchMean, outputSize.C * sizeof(float)));
            checkCuda(cudaMalloc(&d_batchInvVariance, outputSize.C * sizeof(float)));
        }
        else if (backON && !gradON) {
            backON = false;
            checkCuda(cudaFree(d_backOutputTensor));
            checkCuda(cudaFree(d_scaleGrads));
            checkCuda(cudaFree(d_shiftGrads));
            checkCuda(cudaFree(d_batchMean));
            checkCuda(cudaFree(d_batchInvVariance));
            d_backOutputTensor = nullptr;
            d_scaleGrads = nullptr;
            d_shiftGrads = nullptr;
            d_batchMean = nullptr;
            d_batchInvVariance = nullptr;
        }
    };

    void forward(float* d_borrowTensor, const size_t batchSize) override {
        d_prevTensor = d_borrowTensor;
        if (currBatchSize != batchSize) {
            currBatchSize = batchSize;
            checkCuda(cudaFree(d_outputTensor));
            checkCuda(cudaMalloc(&d_outputTensor, batchSize * outputSize.fullSize() * sizeof(float)));
            checkCudnn(cudnnSetTensor4dDescriptor(
                cudnnFullTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                batchSize, outputSize.C, outputSize.H, outputSize.W
            ));
            if (backON) {
                checkCuda(cudaFree(d_backOutputTensor));
                checkCuda(cudaMalloc(&d_backOutputTensor, batchSize * inputSize.fullSize() * sizeof(float)));
            }
        }
        const float alpha = 1.0f;
        const float beta = 0.0f;
        if (backON) {
            checkCudnn(cudnnBatchNormalizationForwardTraining(
                cudnnHandle, CUDNN_BATCHNORM_SPATIAL, &alpha, &beta,
                cudnnFullTensorDesc, d_borrowTensor,
                cudnnFullTensorDesc, d_outputTensor,
                cudnnChannelFlatTensorDesc,
                d_scale, d_shift,
                expAvgFactor, d_runningMean, d_runningVar,
                epsilon, d_batchMean, d_batchInvVariance
            ));
        }
        else {
            checkCudnn(cudnnBatchNormalizationForwardInference(
                cudnnHandle, CUDNN_BATCHNORM_SPATIAL, &alpha, &beta,
                cudnnFullTensorDesc, d_borrowTensor,
                cudnnFullTensorDesc, d_outputTensor,
                cudnnChannelFlatTensorDesc,
                d_scale, d_shift,
                d_runningMean, d_runningVar,
                epsilon
            ));
        }
        next->forward(d_outputTensor, batchSize);
    }

    void backward(float* d_replaceTensor) override {
        d_outputTensor = d_replaceTensor;
        const float alpha = 1.0f;
        const float beta = 0.0f;
        checkCudnn(cudnnBatchNormalizationBackward(
            cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
            &alpha, &beta, &alpha, &beta,
            cudnnFullTensorDesc, d_prevTensor,
            cudnnFullTensorDesc, d_outputTensor,
            cudnnFullTensorDesc, d_backOutputTensor,
            cudnnChannelFlatTensorDesc,
            d_scale, d_scaleGrads, d_shiftGrads,
            epsilon, d_batchMean, d_batchInvVariance
        ));
        prev->backward(d_backOutputTensor);
        d_backOutputTensor = d_prevTensor;
    }

    ~BatchNormLayer() {
        checkCuda(cudaFree(d_outputTensor));
        checkCuda(cudaFree(d_backOutputTensor));
        checkCuda(cudaFree(d_scale));
        checkCuda(cudaFree(d_shift));
        checkCuda(cudaFree(d_runningMean));
        checkCuda(cudaFree(d_runningVar));
        checkCuda(cudaFree(d_scaleGrads));
        checkCuda(cudaFree(d_shiftGrads));
        checkCuda(cudaFree(d_batchMean));
        checkCuda(cudaFree(d_batchInvVariance));
        checkCudnn(cudnnDestroyTensorDescriptor(cudnnFullTensorDesc));
        checkCudnn(cudnnDestroyTensorDescriptor(cudnnChannelFlatTensorDesc));
    }
};




// Refac:
//  Scholastic depth self skip?
//  Multiple next methods?
//  differentate borrowed from owned for clarity

// Optimizer and optimizer types
// Finish batch norm
// Softmax (cuDNN)


// [LRNABLE] Conv2D (cuDNN)
// [LRNABLE] Conv2DDepthwise (Own kernels)

// [LRNABLE] Linear (own / cuBLASS)
// LRNABLE saving