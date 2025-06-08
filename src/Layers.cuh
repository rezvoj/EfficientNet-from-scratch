#include <stdexcept>
#include <random>
#include <cudnn_v9.h>
#include "Kernels.cuh"
#include "Utils.cuh"

constexpr size_t BLOCK_SIZE = 256;
constexpr size_t BLOCK_X_SIZE = 16;
constexpr size_t BLOCK_Y_SIZE = 16;



struct TensorSize {
    size_t ZSize = 1;
    size_t YSize = 1;
    size_t XSize = 1;
    
    size_t fullSize() const {
        return ZSize * YSize * XSize;
    }

    bool operator==(const TensorSize& other) const {
        return ZSize == other.ZSize
                && YSize == other.YSize
                && XSize == other.XSize;
    }

    bool operator!=(const TensorSize& other) const {
        return !(*this == other);
    }
};



class Optimizer {



};



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

    /**
     * @brief Sets the preceding layer in the network graph.
     * @param prevLayer A pointer to the layer that comes before this one.
     */
    void _NNsetPrev(Layer* prevLayer) { prev = prevLayer; }

    /**
     * @brief Virtual destructor.
     * 
     * @note Memory Contract: The destructor of any derived Layer is responsible for freeing
     *       any GPU memory buffer that the layer instance OWNS AT THE MOMENT OF DESTRUCTION.
     *       Due to the memory exchange contract, this may not be the same buffer the
     *       layer originally allocated.
     * 
     * @warning Contract Violation: Under the "Memory Exchange" contract (i.e., when training
     *          with gradients enabled), it is illegal to destroy a layer after its forward()
     *          pass has been called but before its corresponding backward() pass has completed.
     *          Doing so will break the ownership chain and result in memory corruption or a crash.
     */
    virtual ~Layer() {}

    /**
     * @brief Connects this layer to the next layer in the network graph.
     * @param nextLayer A pointer to the layer that will follow this one.
     */
    virtual void append(Layer* nextLayer) {
        if (nextLayer->_NNgetInputSize() != outputSize) {
            throw NNException("Invalid layer input tensor size.");
        }
        if (next == nullptr) next = nextLayer;
        else throw NNException("Layer already connected.");
        nextLayer->_NNsetPrev(this);
    }

    /**
     * @brief Toggles gradient calculation and memory model for training vs. inference.
     * @param gradON If true, enables gradient calculations and the "Memory Exchange" ownership model.
     *               If false, disables gradients and uses a "Memory Borrowing" model.
     * 
     * @note Default State & Usage Rules:
     *       The default gradient state for any new Layer is Off (inference mode).
     *       Calling the `backward()` method is illegal when the gradient state is Off.
     * 
     * @warning Contract Violation: It is illegal to call this function after a forward() pass
     *          has been initiated and before its corresponding backward() pass has completed.
     *          Changing the memory model mid-operation will break the ownership chain and
     *          result in undefined behavior, memory corruption, or a crash. This function
     *          should only be called between training steps or before inference begins.
     */
    virtual void _NNtoggleGrad([[maybe_unused]] const bool gradON) {};

    /**
     * @brief Performs the forward propagation of data through the layer.
     * @param d_inputTensor Pointer to the input tensor data on the GPU.
     * @param batchSize The number of items in the current batch.
     *
     * @note Memory Contract (Training - Grads ON):
     *       This layer TAKES OWNERSHIP of the memory pointed to by d_inputTensor.
     *       It is now responsible for managing this memory buffer. It will either pass
     *       ownership of this same buffer to the next layer or pass ownership of a new
     *       buffer it creates.
     *
     * @note Memory Contract (Inference - Grads OFF):
     *       This layer BORROWS the memory pointed to by d_inputTensor. It does not
     *       take ownership and must not free it. It will perform its calculations
     *       and pass a borrowed pointer to the next layer.
     */
    virtual void _NNforward(float* d_inputTensor, const size_t batchSize) = 0;

    /**
     * @brief Performs the backward propagation of gradients through the layer.
     * @param d_gradientTensor Pointer to the gradient tensor from the subsequent layer.
     *
     * @note Memory Contract (Training only):
     *       This layer TAKES OWNERSHIP of the memory pointed to by d_gradientTensor.
     *       This completes the "memory exchange" initiated in the forward pass. The layer
     *       is now responsible for this buffer. It will then perform its gradient
     *       calculations and pass ownership of a gradient buffer to its previous layer
     *       via its own backward() call.
     *
     * @note State Dependency: This method implicitly depends on the state (e.g., batchSize)
     *       set by the most recent forward() call. It is invalid to call backward()
     *       without a preceding forward() call within the same training step.
     */
    virtual void _NNbackward(float* d_gradientTensor) = 0;

    /**
     * @brief Applies an optimization algorithm to update layer parameters.
     * @param optimizer The optimizer object containing the update logic.
     */
    virtual void _NNoptimize(const Optimizer& optimizer) {
        if (next == nullptr) throw NNException("Layer needs to be connected.");
        next->_NNoptimize(optimizer);
    }

    /**
     * @brief Gets the expected tensor size for a single item (excluding the batch dimension).
     * @return A TensorSize struct describing the C, H, W dimensions.
     */
    virtual TensorSize _NNgetInputSize() const { return inputSize; }

};



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

    // Must be appended in the same order as the corresponding merge is to the paths
    void append(Layer* nextLayer) override {
        if (nextLayer->_NNgetInputSize() != outputSize) {
            throw NNException("Invalid layer input tensor size.");
        }
        if (next == nullptr) next = nextLayer;
        else if (next2 == nullptr) next2 = nextLayer;
        else throw NNException("Layer already connected.");
        nextLayer->_NNsetPrev(this);
    }

    void _NNforward(float* d_borrowTensor, const size_t batchSize) override {
        if (next2 == nullptr || next2 == nullptr) {
            throw NNException("Layer needs to be connected.");
        }
        if (currBatchSize != batchSize) {
            currBatchSize = batchSize;
            checkCuda(cudaFree(d_copyTensor));
            checkCuda(cudaMalloc(&d_copyTensor, batchSize * inputSize.fullSize() * sizeof(float)));
        }
        const size_t copySize = currBatchSize * inputSize.fullSize() * sizeof(float);
        checkCuda(cudaMemcpy(d_copyTensor, d_borrowTensor, copySize, cudaMemcpyDeviceToDevice));
        next->_NNforward(d_borrowTensor, currBatchSize);
        next2->_NNforward(d_copyTensor, currBatchSize);
    }

    void _NNbackward(float* d_replaceTensor) override {
        if (prev == nullptr) {
            throw NNException("Layer needs to be connected.");
        }
        if (!partialBackward) {
            d_copyTensor = d_replaceTensor;
            partialBackward = true;
            return;
        }
        partialBackward = false;
        const size_t fullSize = currBatchSize * inputSize.fullSize();
        elementwiseAddInplace<<<ceilDiv(fullSize, BLOCK_SIZE), BLOCK_SIZE>>>(
            d_replaceTensor, d_copyTensor, fullSize
        ); checkCudaLastError();
        prev->_NNbackward(d_replaceTensor);
    }

    void _NNoptimize(const Optimizer& optimizer) override {
        if (next == nullptr || next2 == nullptr) {
            throw NNException("Layer needs to be connected.");
        }
        next->_NNoptimize(optimizer);
        next2->_NNoptimize(optimizer);
    }

    ~SplitLayer() override {
        checkCuda(cudaFree(d_copyTensor));
    }
};



class MulMergeLayer : public Layer {
private:
    float* d_outTensor;
    Layer* prev2;
    float* d_prev1Tensor;
    float* d_prev2Tensor;
    bool partialForward;

public:
    MulMergeLayer(const TensorSize inputSize): 
            Layer(inputSize, inputSize),
            d_outTensor(nullptr),
            prev2(nullptr),
            d_prev1Tensor(nullptr),
            d_prev2Tensor(nullptr),
            partialForward(false) {}

    void _NNforward(float* d_borrowTensor, const size_t batchSize) override {
        if (next == nullptr) throw NNException("Layer needs to be connected.");
        if (!partialForward) {
            if (currBatchSize != batchSize) {
                currBatchSize = batchSize;
                checkCuda(cudaFree(d_outTensor));
                checkCuda(cudaMalloc(&d_outTensor, batchSize * inputSize.fullSize() * sizeof(float)));
            }
            d_prev1Tensor = d_borrowTensor;
            return;
        }
        d_prev2Tensor = d_borrowTensor;
        const size_t fullSize = currBatchSize * inputSize.fullSize();
        elementwiseMul<<<ceilDiv(fullSize, BLOCK_SIZE), BLOCK_SIZE>>>(
            d_outTensor, d_prev1Tensor, d_prev2Tensor, fullSize
        ); checkCudaLastError();
        next->_NNforward(d_outTensor, currBatchSize);
    }

    void _NNbackward(float* d_replaceTensor) override {
        if (prev == nullptr || prev2 == nullptr) {
            throw NNException("Layer needs to be connected.");
        }
        d_outTensor = d_replaceTensor;
        const size_t fullSize = currBatchSize * inputSize.fullSize();
        elementwiseMulBackwardInplace<<<ceilDiv(fullSize, BLOCK_SIZE), BLOCK_SIZE>>>(
            d_prev2Tensor, d_prev1Tensor, d_outTensor, fullSize
        ); checkCudaLastError();
        prev2->_NNbackward(d_prev2Tensor);
        prev->_NNbackward(d_prev1Tensor);
        d_prev2Tensor = nullptr;
        d_prev1Tensor = nullptr;
    }

    ~MulMergeLayer() override {
        checkCuda(cudaFree(d_outTensor));
    }
};



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

    void _NNforward(float* d_borrowTensor, const size_t batchSize) override {
        if (next == nullptr) throw NNException("Layer needs to be connected.");
        if (!partialForward) {
            currBatchSize = batchSize;
            d_prevTensor = d_borrowTensor;
            return;
        }
        const size_t fullSize = currBatchSize * inputSize.fullSize();
        elementwiseAddInplace<<<ceilDiv(fullSize, BLOCK_SIZE), BLOCK_SIZE>>>(
            d_borrowTensor, d_prevTensor, fullSize
        ); checkCudaLastError();
        next->_NNforward(d_borrowTensor, currBatchSize);
    }

    void _NNbackward(float* d_replaceTensor) override {
        if (prev == nullptr || prev2 == nullptr) {
            throw NNException("Layer needs to be connected.");
        }
        const size_t copySize = currBatchSize * inputSize.fullSize() * sizeof(float);        
        checkCuda(cudaMemcpy(d_prevTensor, d_replaceTensor, copySize, cudaMemcpyDeviceToDevice));
        prev2->_NNbackward(d_replaceTensor);
        prev->_NNbackward(d_prevTensor);
        d_prevTensor = nullptr;
    }
};



class StochasticDepthLayer : public Layer {
private:
    bool skipON;
    bool skippedForward;
    float retainRate;
    std::mt19937 randomGenerator;
    std::uniform_real_distribution<> distribution;

public:
    StochasticDepthLayer(const TensorSize inputSize, const float rate): 
            Layer(inputSize, inputSize),
            skipON(false),
            skippedForward(false),
            retainRate(rate) {
        std::random_device device;
        randomGenerator = std::mt19937(device());
        distribution = std::uniform_real_distribution<>(0.0, 1.0);
    }

    void _NNtoggleGrad(const bool gradON) override { skipON = gradON; };

    void _NNforward(float* d_borrowTensor, const size_t batchSize) override {
        if (next == nullptr) throw NNException("Layer needs to be connected.");
        float multiplier = 1.0;
        if (!skipON) multiplier = retainRate;
        else if (distribution(randomGenerator) < static_cast<double>(retainRate)) {
            multiplier = 0.0f;
            skippedForward = true;
        }
        if (multiplier != 1.0f) {
            const size_t fullSize = currBatchSize * inputSize.fullSize();
            elementwiseScalarMulInplace<<<ceilDiv(fullSize, BLOCK_SIZE), BLOCK_SIZE>>>(
                d_borrowTensor, retainRate, fullSize
            ); checkCudaLastError();
        }    
        next->_NNforward(d_borrowTensor, currBatchSize);
    }

    void _NNbackward(float* d_replaceTensor) override {
        if (prev == nullptr) throw NNException("Layer needs to be connected.");
        if (skippedForward) {
            skippedForward = false;
            const size_t fullSize = currBatchSize * inputSize.fullSize();
            elementwiseScalarMulInplace<<<ceilDiv(fullSize, BLOCK_SIZE), BLOCK_SIZE>>>(
                d_replaceTensor, 0.0f, fullSize
            ); checkCudaLastError();
        }
        prev->_NNbackward(d_replaceTensor);
    }
};



class DropoutLayer : public Layer {
private:
    bool dropON;
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
                dropON(false),
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

    void _NNtoggleGrad(const bool gradON) override {
        if (!dropON && gradON) {
            dropON = true;
            checkCuda(cudaMalloc(&d_cudnnDropoutStates, cudnnDropoutStatesSize));
            checkCudnn(cudnnSetDropoutDescriptor(
                cudnnDropoutDesc, cudnnHandle, dropoutRate,
                d_cudnnDropoutStates, cudnnDropoutStatesSize, dropoutSeed
            ));
        }
        else if (dropON && !gradON) {
            dropON = false;
            checkCuda(cudaFree(d_cudnnDropoutStates));
            checkCuda(cudaFree(d_cudnnReserveSpace));
            d_cudnnDropoutStates = nullptr;
            d_cudnnReserveSpace = nullptr;
            cudnnReserveSpaceSize = 0;
            currBatchSize = 0;
        }
    };

    void _NNforward(float* d_borrowTensor, const size_t batchSize) override {
        if (next == nullptr) throw NNException("Layer needs to be connected.");
        if (dropON) {
            if (currBatchSize != batchSize) {
                currBatchSize = batchSize;
                checkCudnn(cudnnSetTensor4dDescriptor(
                    cudnnTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                    currBatchSize, inputSize.ZSize, inputSize.YSize, inputSize.XSize
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
        next->_NNforward(d_borrowTensor, batchSize);
    }

    void _NNbackward(float* d_replaceTensor) override {
        if (prev == nullptr) throw NNException("Layer needs to be connected.");
        checkCudnn(cudnnDropoutBackward(
            cudnnHandle, cudnnDropoutDesc,
            cudnnTensorDesc, d_replaceTensor,
            cudnnTensorDesc, d_replaceTensor,   
            d_cudnnReserveSpace, cudnnReserveSpaceSize
        ));
        prev->_NNbackward(d_replaceTensor);
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
    bool keepInputON;
    float* d_Tensor;

public:
    ActivationLayer(const TensorSize inputSize): 
            Layer(inputSize, inputSize),
            keepInputON(false),
            d_Tensor(nullptr) {}
    
    void _NNtoggleGrad(const bool gradON) override {
        if (!keepInputON && gradON) keepInputON = true;
        else if (keepInputON && !gradON) {
            keepInputON = false;
            checkCuda(cudaFree(d_Tensor));
            d_Tensor = nullptr;
            currBatchSize = 0;
        }
    };

    void _NNforward(float* d_borrowTensor, const size_t batchSize) override {
        if (next == nullptr) throw NNException("Layer needs to be connected.");
        if (keepInputON) {
            if (currBatchSize != batchSize) {
                currBatchSize = batchSize;
                checkCuda(cudaFree(d_Tensor));
                checkCuda(cudaMalloc(&d_Tensor, batchSize * inputSize.fullSize() * sizeof(float)));
            }
            const size_t fullSize = currBatchSize * inputSize.fullSize();
            elementwiseActivation<ACTIVATION>
                <<<ceilDiv(fullSize, BLOCK_SIZE), BLOCK_SIZE>>>(d_Tensor, d_borrowTensor, fullSize); 
            checkCudaLastError();
            std::swap(d_borrowTensor, d_Tensor);
        }
        else {
            const size_t fullSize = currBatchSize * inputSize.fullSize();
            elementwiseActivationInplace<ACTIVATION>
                <<<ceilDiv(fullSize, BLOCK_SIZE), BLOCK_SIZE>>>(d_borrowTensor, fullSize);
            checkCudaLastError();
        }
        next->_NNforward(d_borrowTensor, batchSize);
    }

    void _NNbackward(float* d_replaceTensor) override {
        if (prev == nullptr) throw NNException("Layer needs to be connected.");
        const size_t fullSize = currBatchSize * inputSize.fullSize();
        elementwiseActivationBackwardInplace<ACTIVATION>
            <<<ceilDiv(fullSize, BLOCK_SIZE), BLOCK_SIZE>>>(d_replaceTensor, d_Tensor, fullSize); 
        checkCudaLastError();
        prev->_NNbackward(d_replaceTensor);
    }

    ~ActivationLayer() override {
        checkCuda(cudaFree(d_Tensor));
    }
};








// Conv2D (cuDNN)
// Conv2DDepthwise (Own kernels)
// Linear (own / cuBLASS)

// Softmax (cuDNN)
// BatchNorm (cuDNN)

// Pooling / Expansion (Own Kernels)
