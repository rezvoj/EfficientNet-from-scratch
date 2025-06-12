#include <exception>
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



class Layer {
protected:
    bool backOn;
    TensorSize inputSize;
    TensorSize outputSize;
    size_t currBatchSize;
    size_t currActualBatchSize;
    Layer *prev, *next;

public:
    // The instance is meant to be allocated dynamically 
    Layer(const TensorSize inputSize, const TensorSize outputSize):
            backOn(false),        
            inputSize(inputSize),
            outputSize(outputSize),
            currBatchSize(0),
            currActualBatchSize(0),
            prev(nullptr),
            next(nullptr) {}
    
    virtual void _setPrev(Layer* prevLayer) { prev = prevLayer; }
    virtual void _setNext(Layer* nextLayer) { next = nextLayer; }
    
    // The layer frees the resources and buffers which it currently owns
    // During training it can exchnage the ownership of buffers with next and previous layers
    virtual ~Layer() {}
    
    // Merhod to connect the layers with each other
    // Require the first layer's output size and second layer's input size to be the same
    virtual void connect(Layer* nextLayer) {
        next = nextLayer;
        nextLayer->_setPrev(this);
    }
    
    // Changes the layer from training to inference mode and vice versa
    // Allocating and freeing buffers which are / are not needed
    virtual void toggleTrain([[maybe_unused]] const bool trainOn) {
        if (backOn != trainOn) {
            backOn = trainOn;
            // Reset the batch sizes to force buffer reallocations
            currBatchSize = 0;
            currActualBatchSize = 0;
        }
    };
    
    // Performs forward pass calculation for the current layer
    // Potentially saves the borrowed tensor to non owning pointer
    // Borrows it's output tensor with the activation data to the next layer
    virtual void forward(float* d_inputTensor, const size_t batchSize) = 0;
    
    // Performs backwards pass calculation for the current layer
    // Changes the memory buffer borrowing to memory trade contract
    // It claims the memory borrowed during forward pass from previous layer
    // Gives different piece of memory of the same size to the previous layer
    // which contains the calculated gradient values entering the previous layer
    virtual void backward(float* d_gradientTensor) = 0;
    
    // Performs a single optimizer step on the current layer escalates it onto next layer
    virtual void optimize(const Optimizer& optimizer) { next->optimize(optimizer); }
};



class SplitLayer : public Layer {
private:
    bool partialBackward;
    bool stochasticSkip;
    float retainRate;
    Layer* nextResidual;
    Layer* nextMerge;
    float* d_oCopyTensor;
    std::mt19937* randomGenerator;
    std::uniform_real_distribution<> distribution;

public:
    SplitLayer(const TensorSize inputSize, const float retainRate):
            Layer(inputSize, inputSize),
            partialBackward(false),
            stochasticSkip(false),
            retainRate(retainRate),
            nextResidual(nullptr),
            nextMerge(nullptr),
            d_oCopyTensor(nullptr),
            randomGenerator(nullptr) {
        if (retainRate < 1.0f) {
            std::random_device device;
            randomGenerator = new std::mt19937(device());
            distribution = std::uniform_real_distribution<>(0.0, 1.0);
        }
    }


    // Connect to the merge layer for skipping the non residual path
    void _setNextMerge(Layer* nextMergeLayer) { nextMerge = nextMergeLayer; }


    // Connect the residual path's layer
    void connectResidual(Layer* nextResidualLayer) {
        if (nextResidual == nullptr) nextResidual = nextResidualLayer;
        nextResidualLayer->_setPrev(this);
    }


    void forward(float* d_inputTensor, const size_t batchSize) override {
        const size_t fullSize = batchSize * inputSize.fullSize();
        // Save the batch size for backwards pass
        currBatchSize = batchSize;
        // Reallocate the buffers if batch size was changed
        if (currActualBatchSize < batchSize) {
            currActualBatchSize = batchSize;
            checkCuda(cudaFree(d_oCopyTensor));
            checkCuda(cudaMalloc(&d_oCopyTensor, fullSize * sizeof(float)));
        }
        // Case for inference
        if (!backOn) {
            // Scale the input tensor to the retain rate
            if (retainRate != 1.0f) {
                elementwiseScalarMulInplace<<<ceilDiv(fullSize, BLOCK_SIZE), BLOCK_SIZE>>>(
                    d_inputTensor, retainRate, fullSize
                );
                checkCudaLastError();
            }
            // Copy out the input tensor into 2 tensor for the splitting
            checkCuda(cudaMemcpy(d_oCopyTensor, d_inputTensor, fullSize * sizeof(float), cudaMemcpyDeviceToDevice));
            // Lend the both tensors to the next layers
            // The correct the correct order (forward to residual as second) is crucial there
            next->forward(d_oCopyTensor, batchSize);
            nextResidual->forward(d_inputTensor, batchSize);
        }
        // Case for training without skip
        else if (retainRate >= 1.0f || distribution(randomGenerator) <= static_cast<double>(retainRate)) {
            // Copy out the input tensor into 2 tensor for the splitting
            checkCuda(cudaMemcpy(d_oCopyTensor, d_inputTensor, fullSize * sizeof(float), cudaMemcpyDeviceToDevice));
            // Lend the both tensors to the next layers
            // The correct the correct order (forward to residual as second) is crucial there
            next->forward(d_oCopyTensor, batchSize);
            nextResidual->forward(d_inputTensor, batchSize);
        }
        // Case for training with skip
        else {
            // Pass nullptr as tensor to the next merge layer (signaling skip)
            // The correct the correct order (forward to residual as second) is crucial there
            nextMerge->forward(nullptr, batchSize);
            // Lend the input tensor to next residual path layer
            nextResidual->forward(d_inputTensor, batchSize);
        }
    }


    void backward(float* d_gradientTensor) override {
        // Reclaim the incoming gradient tensor from non residual connection
        if (!partialBackward) {
            partialBackward = true;
            // Reclaim the incoming tensor or keep the current one based on SD skip
            if (d_gradientTensor == nullptr) stochasticSkip = true;
            else d_oCopyTensor = d_gradientTensor;
            return;
        }
        // Conditionally add the gradient tensor from non residual into the residual path's tensor
        if (!stochasticSkip) {
            const size_t fullSize = currBatchSize * inputSize.fullSize();
            elementwiseAddInplace<<<ceilDiv(fullSize, BLOCK_SIZE), BLOCK_SIZE>>>(
                d_gradientTensor, d_oCopyTensor, fullSize
            );
            checkCudaLastError();
        }
        // Reset the partial backward and stochastic depth skip flags
        partialBackward = false;
        stochasticSkip = false;
        // Complete the trade by giving back the tensor with gradient data
        prev->backward(d_gradientTensor);
    }


    void optimize(const Optimizer& optimizer) override {
        next->optimize(optimizer);
        nextResidual->optimize(optimizer);
    }


    ~SplitLayer() override {
        if (!std::uncaught_exceptions()) {
            checkCuda(cudaFree(d_oCopyTensor));
            delete randomGenerator;
        }
    }

};



class AddMergeLayer : public Layer {
private:
    bool partialForward;
    bool stochasticSkip;
    Layer* prevResidual;
    SplitLayer* prevSplit;
    float* d_bPrevTensor;

public:
    AddMergeLayer(const TensorSize inputSize): 
            Layer(inputSize, inputSize),
            partialForward(false),
            stochasticSkip(false),
            prevResidual(nullptr),
            prevSplit(nullptr),
            d_bPrevTensor(nullptr) {}


    // Connect to the previous layers with inverted approach instead to avoid ambiguity
    virtual void _setPrev(Layer* prevLayer) {}


    // Connect the non residual incoming path
    void setPrev(Layer* prevLayer) {
        prevLayer->_setNext(this);
        prev = prevLayer;
    }


    // Connect the residual incoming path
    void setPrevResidual(Layer* prevResidualLayer) {
        prevResidualLayer->_setNext(this);
        prevResidual = prevResidualLayer;
    }


    // Connect the corresponding split layer to conditionally skip non residual path
    void setPrevSplit(SplitLayer* prevSplitLayer) {
        prevSplitLayer->_setNextMerge(this);
        prevSplit = prevSplitLayer;
    }


    void forward(float* d_inputTensor, const size_t batchSize) override {
        // Save the batch size for backwards pass
        currBatchSize = batchSize;
        // Handle forward pass comming from the non residual connection
        if (!partialForward) {
            partialForward = true;
            // Borrow the input tensor and conditionally set the SD flag
            d_bPrevTensor = d_inputTensor;
            stochasticSkip = (d_inputTensor == nullptr);
            return;
        }
        // Handle forward pass comming from the residual connection
        partialForward = false;
        // Skip the addition and forward just the tensor from residual connection
        if (!stochasticSkip) {
            // Add the input tensors into the non residual's input tensor
            const size_t fullSize = batchSize * inputSize.fullSize();
            elementwiseAddInplace<<<ceilDiv(fullSize, BLOCK_SIZE), BLOCK_SIZE>>>(
                d_inputTensor, d_bPrevTensor, fullSize
            );
            checkCudaLastError();
        }
        // Lend the borrowed residual's input tensor with activation data to next layer
        next->forward(d_inputTensor, batchSize);
    }


    void backward(float* d_gradientTensor) override {
        // Conditionally skip backprob through the non residual path
        if (!stochasticSkip) {
            const size_t copySize = currBatchSize * inputSize.fullSize() * sizeof(float);
            // Copy out the gradients for the non residual path into it's input tensor
            checkCuda(cudaMemcpy(d_bPrevTensor, d_gradientTensor, copySize, cudaMemcpyDeviceToDevice));
            // Complete the trade by giving back the non residual's tensor with gradient data
            prev->backward(d_bPrevTensor);
        }
        // Or signal the skip for the split layer
        else prevSplit->backward(nullptr);
        // Reset the partial forward and stochastic skip flags
        partialForward = false;
        stochasticSkip = false;
        // Complete the trade by giving back the residual's tensor with gradient data
        // The correct the correct order (backward to residual as second) is crucial there
        prevResidual->backward(d_gradientTensor);
    }

};



class MulMergeLayer : public Layer {
private:
    bool partialForward;
    bool stochasticSkip;
    Layer* prevResidual;
    SplitLayer* prevSplit;
    float* d_oOutTensor;
    float* d_bPrevTensor;
    float* d_bPrevResidualTensor;

public:
    MulMergeLayer(const TensorSize inputSize): 
            Layer(inputSize, inputSize),
            partialForward(false),
            stochasticSkip(false),
            prevResidual(nullptr),
            prevSplit(nullptr),
            d_oOutTensor(nullptr),
            d_bPrevTensor(nullptr),
            d_bPrevResidualTensor(nullptr) {}


    // Connect to the previous layers with inverted approach instead to avoid ambiguity
    virtual void _setPrev(Layer* prevLayer) {}


    // Connect the non residual incoming path
    void setPrev(Layer* prevLayer) {
        prevLayer->_setNext(this);
        prev = prevLayer;
    }


    // Connect the residual incoming path
    void setPrevResidual(Layer* prevResidualLayer) {
        prevResidualLayer->_setNext(this);
        prevResidual = prevResidualLayer;
    }


    // Connect the corresponding split layer to conditionally skip non residual path
    void setPrevSplit(SplitLayer* prevSplitLayer) {
        prevSplitLayer->_setNextMerge(this);
        prevSplit = prevSplitLayer;
    }


    void toggleTrain(const bool trainOn) override {
        if (backOn != trainOn) {
            backOn = trainOn;
            // Reset the batch sizes to force buffer reallocations
            currBatchSize = 0;
            currActualBatchSize = 0;
            // Free the unnecessary memory for inference
            if (!trainOn) {
                checkCuda(cudaFree(d_oOutTensor));
                d_oOutTensor = nullptr;
            }
        }
    };


    void forward(float* d_inputTensor, const size_t batchSize) override {
        const size_t fullSize = batchSize * inputSize.fullSize();
        // Save the batch size for backwards pass
        currBatchSize = batchSize;
        // Handle forward pass comming from the non residual connection
        if (!partialForward) {
            partialForward = true;
            // Resize the output tensor if needed
            if (backOn && currActualBatchSize < batchSize) {
                currActualBatchSize = batchSize;
                checkCuda(cudaFree(d_oOutTensor));
                checkCuda(cudaMalloc(&d_oOutTensor, fullSize * sizeof(float)));
            }
            // Borrow the input tensor and conditionally set the SD flag
            d_bPrevTensor = d_inputTensor;
            stochasticSkip = (d_inputTensor == nullptr);
            return;
        }
        // Handle forward pass comming from the residual connection
        partialForward = false;
        // Borrow the input tensor as the residual path's tensor
        d_bPrevResidualTensor = d_inputTensor;
        // Case for inference
        if (!backOn) {
            // Multiply the input tensors into the residual's input tensor
            elementwiseMulInplace<<<ceilDiv(fullSize, BLOCK_SIZE), BLOCK_SIZE>>>(
                d_bPrevResidualTensor, d_bPrevTensor, fullSize
            );
            checkCudaLastError();
            // Lend the borrowed residual's input tensor with activation data to next layer
            next->forward(d_bPrevResidualTensor, batchSize);
        }
        // Case for training without skip
        else if (backOn) {   
            // Multiply the input tensors into the output tensor
            elementwiseMul<<<ceilDiv(fullSize, BLOCK_SIZE), BLOCK_SIZE>>>(
                d_oOutTensor, d_bPrevTensor, d_bPrevResidualTensor, fullSize
            );
            checkCudaLastError();
            // Lend the output tensor with activation data to next layer
            next->forward(d_oOutTensor, batchSize);
        }
        // Case for training without skip
        else {
            // Lend the borrowed residual's input tensor with it's data to next layer
            next->forward(d_bPrevResidualTensor, batchSize);
        }
    }


    void backward(float* d_gradientTensor) override {
        // Case without skip
        if (!stochasticSkip) {
            // Reclaim the incoming gradient tensor as owned output tensor
            d_oOutTensor = d_gradientTensor;
            // Calculate the gradients for both paths into their input tensors
            const size_t fullSize = currBatchSize * inputSize.fullSize();
            elementwiseMulBackwardInplace<<<ceilDiv(fullSize, BLOCK_SIZE), BLOCK_SIZE>>>(
                d_bPrevTensor, d_bPrevResidualTensor, d_oOutTensor, fullSize
            );
            checkCudaLastError();
            // Complete the trade by giving back the tensors with gradient data
            // The correct the correct order (backward to residual as second) is crucial there
            prev->backward(d_bPrevTensor);
            prevResidual->backward(d_bPrevResidualTensor);
        }
        // Case with skip
        else {
            // Signal the skip for the split layer
            prevSplit->backward(nullptr);
            // Complete the trade by giving back the residual's tensor with gradient data
            // The correct the correct order (backward to residual as second) is crucial there
            // The borrowed residual's tensor was given to this layer instead of owned tensor
            prevResidual->backward(d_gradientTensor);
        }
        // Reset the partial forward and stochastic skip flags
        partialForward = false;
        stochasticSkip = false;
    }


    ~MulMergeLayer() override {
        if (!std::uncaught_exceptions()) {
            checkCuda(cudaFree(d_oOutTensor));
        }
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
