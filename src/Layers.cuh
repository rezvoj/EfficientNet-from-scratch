#include <stdexcept>
#include <cudnn_v9.h>
#include "Kernels.cuh"
#include "Utils.cuh"

constexpr uint BLOCK_SIZE = 256;
constexpr uint BLOCK_X_SIZE = 16;
constexpr uint BLOCK_Y_SIZE = 16;



class NNException : public std::runtime_error {
    public: NNException(const std::string& errorMessage): 
        std::runtime_error(errorMessage) {}
};



struct TensorSize {
    size_t ZSize = 1;
    size_t YSize = 1;
    size_t XSize = 1;
    
    size_t fullSize() const {
        return ZSize * YSize * XSize * sizeof(float);
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
    Layer *prev;

public:
    Layer() : prev(nullptr) {}

    /**
     * @brief Sets the preceding layer in the network graph.
     * @param prevLayer A pointer to the layer that comes before this one.
     */
    void _setPrev(Layer* prevLayer) { prev = prevLayer; }

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
    virtual void append(Layer* nextLayer) = 0;

    /**
     * @brief Toggles gradient calculation and memory model for training vs. inference.
     * @param gradON If true, enables gradient calculations and the "Memory Exchange" ownership model.
     *               If false, disables gradients and uses a "Memory Borrowing" model.
     */
    virtual void toggleGrad(const bool gradON) = 0;

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
    virtual void forward(float* d_inputTensor, const size_t batchSize) = 0;

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
    virtual void backward(float* d_gradientTensor) = 0;

    /**
     * @brief Applies an optimization algorithm to update layer parameters.
     * @param optimizer The optimizer object containing the update logic.
     */
    virtual void optimize(const Optimizer& optimizer) = 0;

    /**
     * @brief Gets the expected tensor size for a single item (excluding the batch dimension).
     * @return A TensorSize struct describing the C, H, W dimensions.
     */
    virtual TensorSize getInputSize() const = 0;
};



class SplitLayer : public Layer {
private:
    TensorSize tensorSize;
    size_t currBatchSize;
    float* d_copyTensor;
    Layer *next0, *next1;
    bool partialBackward;
    cudnnHandle_t cudnnHandle;
    cudnnTensorDescriptor_t cudnnTensorDesc;
    cudnnOpTensorDescriptor_t cudnnOpDesc;

public:
    SplitLayer(const TensorSize inputSize, const cudnnHandle_t handle): 
            Layer(),
            tensorSize(inputSize),
            currBatchSize(0),
            d_copyTensor(nullptr),
            next0(nullptr),
            next1(nullptr),
            partialBackward(false),
            cudnnHandle(handle) {
        checkCudnn(cudnnCreateTensorDescriptor(&cudnnTensorDesc));
        checkCudnn(cudnnSetTensor4dDescriptor(
            cudnnTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
            1, tensorSize.ZSize, tensorSize.YSize, tensorSize.XSize
        ));
        checkCudnn(cudnnCreateOpTensorDescriptor(&cudnnOpDesc));
        checkCudnn(cudnnSetOpTensorDescriptor(
            cudnnOpDesc,
            CUDNN_OP_TENSOR_ADD,
            CUDNN_DATA_FLOAT,
            CUDNN_NOT_PROPAGATE_NAN
        ));
    }

    void append(Layer* nextLayer) override {
        if (nextLayer->getInputSize() != tensorSize) {
            throw NNException("Invalid layer input tensor size.");
        }
        if (next0 == nullptr) next0 = nextLayer;
        else if (next1 == nullptr) next1 = nextLayer;
        else throw NNException("Layer already connected.");
        nextLayer->_setPrev(this);
    }

    void toggleGrad([[maybe_unused]] const bool gradON) override {};

    void forward(float* d_borrowTensor, const size_t batchSize) override {
        if (next0 == nullptr || next1 == nullptr) {
            throw NNException("Layer needs to be connected.");
        }
        if (currBatchSize != batchSize) {
            checkCuda(cudaFree(d_copyTensor));
            currBatchSize = batchSize;
            checkCuda(cudaMalloc(&d_copyTensor, batchSize * tensorSize.fullSize()));
            checkCudnn(cudnnSetTensor4dDescriptor(
                cudnnTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                currBatchSize, tensorSize.ZSize, tensorSize.YSize, tensorSize.XSize
            ));
        }
        const size_t copySize = currBatchSize * tensorSize.fullSize();
        checkCuda(cudaMemcpy(d_copyTensor, d_borrowTensor, copySize, cudaMemcpyDeviceToDevice));
        next0->forward(d_borrowTensor, currBatchSize);
        next1->forward(d_copyTensor, currBatchSize);
    }

    void backward(float* d_replaceTensor) override {
        if (prev == nullptr) {
            throw NNException("Layer needs to be connected.");
        }
        if (!partialBackward) {
            d_copyTensor = d_replaceTensor;
            partialBackward = true;
            return;
        }
        partialBackward = false;
        const float alpha1 = 1.0f;
        const float alpha2 = 1.0f;
        const float beta = 0.0f;
        checkCudnn(cudnnOpTensor(
            cudnnHandle, cudnnOpDesc,
            &alpha1, cudnnTensorDesc, d_copyTensor,
            &alpha2, cudnnTensorDesc, d_replaceTensor,
            &beta, cudnnTensorDesc, d_replaceTensor
        ));
        prev->backward(d_replaceTensor);
    }

    void optimize(const Optimizer& optimizer) override {
        if (next0 == nullptr || next1 == nullptr) {
            throw NNException("Layer needs to be connected.");
        }
        next0->optimize(optimizer);
        next1->optimize(optimizer);
    }

    TensorSize getInputSize() const override { return tensorSize; }

    ~SplitLayer() override {
        checkCuda(cudaFree(d_copyTensor));
        checkCudnn(cudnnDestroyTensorDescriptor(cudnnTensorDesc));
        checkCudnn(cudnnDestroyOpTensorDescriptor(cudnnOpDesc));
    }
};









// Work in progress
class MulMergeLayer : public Layer {
private:
    TensorSize tensorSize;
    size_t currBatchSize;
    float* d_outTensor;
    Layer *next, *prev2;
    float* d_prev1Tensor;
    float* d_prev2Tensor;
    bool partialForward;
    cudnnHandle_t cudnnHandle;
    cudnnTensorDescriptor_t cudnnTensorDesc;
    cudnnOpTensorDescriptor_t cudnnOpDesc;

public:
    MulMergeLayer(const TensorSize inputSize, const cudnnHandle_t handle): 
            Layer(),
            tensorSize(inputSize),
            currBatchSize(0),
            d_outTensor(nullptr),
            next(nullptr),
            prev2(nullptr),
            d_prev1Tensor(nullptr),
            d_prev2Tensor(nullptr),
            partialForward(false),
            cudnnHandle(handle) {
        checkCudnn(cudnnCreateTensorDescriptor(&cudnnTensorDesc));
        checkCudnn(cudnnSetTensor4dDescriptor(
            cudnnTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
            1, tensorSize.ZSize, tensorSize.YSize, tensorSize.XSize
        ));
        checkCudnn(cudnnCreateOpTensorDescriptor(&cudnnOpDesc));
        checkCudnn(cudnnSetOpTensorDescriptor(
            cudnnOpDesc,
            CUDNN_OP_TENSOR_MUL,
            CUDNN_DATA_FLOAT,
            CUDNN_NOT_PROPAGATE_NAN
        ));
    }

    void append(Layer* nextLayer) override {
        if (nextLayer->getInputSize() != tensorSize) {
            throw NNException("Invalid layer input tensor size.");
        }
        if (next0 == nullptr) next0 = nextLayer;
        else if (next1 == nullptr) next1 = nextLayer;
        else throw NNException("Layer already connected.");
        nextLayer->_setPrev(this);
    }

    void toggleGrad([[maybe_unused]] const bool gradON) override {};

    void forward(float* d_borrowTensor, const size_t batchSize) override {
        if (next0 == nullptr || next1 == nullptr) {
            throw NNException("Layer needs to be connected.");
        }
        if (currBatchSize != batchSize) {
            checkCuda(cudaFree(d_copyTensor));
            currBatchSize = batchSize;
            checkCuda(cudaMalloc(&d_copyTensor, batchSize * tensorSize.fullSize()));
            checkCudnn(cudnnSetTensor4dDescriptor(
                cudnnTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                currBatchSize, tensorSize.ZSize, tensorSize.YSize, tensorSize.XSize
            ));
        }
        const size_t copySize = currBatchSize * tensorSize.fullSize();
        checkCuda(cudaMemcpy(d_copyTensor, d_borrowTensor, copySize, cudaMemcpyDeviceToDevice));
        next0->forward(d_borrowTensor, currBatchSize);
        next1->forward(d_copyTensor, currBatchSize);
    }

    void backward(float* d_replaceTensor) override {
        if (prev == nullptr) {
            throw NNException("Layer needs to be connected.");
        }
        if (!partialBackward) {
            d_copyTensor = d_replaceTensor;
            partialBackward = true;
            return;
        }
        partialBackward = false;
        const float alpha1 = 1.0f;
        const float alpha2 = 1.0f;
        const float beta = 0.0f;
        checkCudnn(cudnnOpTensor(
            cudnnHandle, cudnnOpDesc,
            &alpha1, cudnnTensorDesc, d_copyTensor,
            &alpha2, cudnnTensorDesc, d_replaceTensor,
            &beta, cudnnTensorDesc, d_replaceTensor
        ));
        prev->backward(d_replaceTensor);
    }

    void optimize(const Optimizer& optimizer) override {
        if (next0 == nullptr || next1 == nullptr) {
            throw NNException("Layer needs to be connected.");
        }
        next0->optimize(optimizer);
        next1->optimize(optimizer);
    }

    TensorSize getInputSize() const override { return tensorSize; }

    ~MergeLayer() override {
        checkCuda(cudaFree(d_copyTensor));
        checkCudnn(cudnnDestroyTensorDescriptor(cudnnTensorDesc));
        checkCudnn(cudnnDestroyOpTensorDescriptor(cudnnOpDesc));
    }
};

