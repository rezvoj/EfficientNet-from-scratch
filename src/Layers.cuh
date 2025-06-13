#include <exception>
#include <random>
#include <vector>
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
private:
    size_t iteration;
    float learningRate;
    float beta1;
    float beta2;
    float weightDecay;
    float epsilon;

private:
    struct AdamWeightData {
        float* d_weightTensor;
        float* d_oMMomentTensor;
        float* d_oVMomentTensor;
        const float* d_batchGradTensor;
        size_t tensorFullSize;

        AdamWeightData(
                float* d_weightTensor,
                const float* d_batchGradTensor,
                size_t tensorFullSize) {
            this->d_weightTensor = d_weightTensor;
            this->d_batchGradTensor = d_batchGradTensor;
            this->tensorFullSize = tensorFullSize; 
            checkCuda(cudaMalloc(&d_oMMomentTensor, tensorFullSize * sizeof(float)));
            checkCuda(cudaMalloc(&d_oVMomentTensor, tensorFullSize * sizeof(float)));
            initValues<<<ceilDiv(tensorFullSize, BLOCK_SIZE), BLOCK_SIZE>>>(d_oMMomentTensor, 0.0f, tensorFullSize);
            checkCudaLastError();
            initValues<<<ceilDiv(tensorFullSize, BLOCK_SIZE), BLOCK_SIZE>>>(d_oVMomentTensor, 0.0f, tensorFullSize);
            checkCudaLastError();
        }        
    };

private:
    std::vector<AdamWeightData> adamParams;
    std::vector<AdamWeightData> adamWParams;

public:
    enum OptimAlgo { ADAM, ADAM_W };

public:
    Optimizer(
            const float learningRate,
            const float beta1,
            const float beta2,
            const float weightDecay,
            const float epsilon):
                iteration(1), 
                learningRate(learningRate),
                beta1(beta1),
                beta2(beta2),
                weightDecay(weightDecay),
                epsilon(epsilon) {}


    // Registers weight tensors for layer to update
    // The registered tensors are assumed to become
    //  invalid when layer switches to inference mode
    //  so the optimizer needs to be reset
    void registerLayer(
            OptimAlgo algorithm,
            float* d_weightTensor,
            const float* d_batchGradTensor,
            const size_t tensorFullSize) {        
        switch (algorithm) {
            case ADAM:
                adamParams.emplace_back(d_weightTensor, d_batchGradTensor, tensorFullSize);
                break;
            case ADAM_W: 
                adamWParams.emplace_back(d_weightTensor, d_batchGradTensor, tensorFullSize);
                break;
        }
    }


    // Updates all the registred weights using their registered alogrithm
    void step(const size_t batchSize) {
        for (AdamWeightData& layerData : adamParams) {
            adamOptimizerStep<<<ceilDiv(layerData.tensorFullSize, BLOCK_SIZE), BLOCK_SIZE>>>(
                layerData.d_weightTensor, layerData.d_oMMomentTensor, layerData.d_oVMomentTensor, 
                layerData.d_batchGradTensor, static_cast<float>(iteration),
                learningRate, beta1, beta2, epsilon,
                static_cast<float>(batchSize), layerData.tensorFullSize
            );
            checkCudaLastError();
        }
        for (AdamWeightData& layerData : adamWParams) {
            adamWOptimizerStep<<<ceilDiv(layerData.tensorFullSize, BLOCK_SIZE), BLOCK_SIZE>>>(
                layerData.d_weightTensor, layerData.d_oMMomentTensor, layerData.d_oVMomentTensor,
                layerData.d_batchGradTensor, static_cast<float>(iteration),
                learningRate, beta1, beta2, weightDecay, epsilon,
                static_cast<float>(batchSize), layerData.tensorFullSize
            );
            checkCudaLastError();
        }
        iteration += 1;
    }


    // Resets the internal state back to state after construction
    void reset() {
        for (AdamWeightData& layerData : adamParams) {
            checkCuda(cudaFree(layerData.d_oMMomentTensor));
            checkCuda(cudaFree(layerData.d_oVMomentTensor));
        }
        for (AdamWeightData& layerData : adamWParams) {
            checkCuda(cudaFree(layerData.d_oMMomentTensor));
            checkCuda(cudaFree(layerData.d_oVMomentTensor));
        }
        adamParams.clear();
        adamWParams.clear();
        iteration = 1;
    };


    ~Optimizer() { reset(); }

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
    //  which contains the calculated gradient values entering the previous layer
    virtual void backward(float* d_gradientTensor) = 0;

    // The layer frees the resources and buffers which it currently owns
    // During training it can exchnage the ownership of buffers with next and previous layers
    virtual ~Layer() {}
};



class LearnableLayer : public Layer {
protected:
    Optimizer::OptimAlgo optimizerAlgorithm;

public:
    LearnableLayer(
            const TensorSize inputSize,
            const TensorSize outputSize,
            const Optimizer::OptimAlgo algorithm):
                Layer(inputSize, outputSize),
                optimizerAlgorithm(algorithm) {}
    
    // Initializes weights to internally spcified values
    virtual void initWeights() = 0;

    // Registers the layer's weight and weight gradient tensors
    // The registered tensors CAN'T be exchanged with other layers in memory trade
    // Doesn't do anything if layer is in inference mode
    virtual void registerWeights(Optimizer& optimizer) = 0;
};



class SplitLayer : public Layer {
private:
    bool partialBackward;
    bool stochasticSkip;
    float retainRate;
    Layer* nextShortcut;
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
            nextShortcut(nullptr),
            nextMerge(nullptr),
            d_oCopyTensor(nullptr),
            randomGenerator(nullptr) {
        if (retainRate < 1.0f) {
            std::random_device device;
            randomGenerator = new std::mt19937(device());
            distribution = std::uniform_real_distribution<>(0.0, 1.0);
        }
    }


    // Connect to the merge layer for skipping the full path
    void _setNextMerge(Layer* nextMergeLayer) { nextMerge = nextMergeLayer; }


    // Connect the shortcut path's layer
    void connectShortcut(Layer* nextShortcutLayer) {
        if (nextShortcut == nullptr) nextShortcut = nextShortcutLayer;
        nextShortcutLayer->_setPrev(this);
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
            // The correct order (forward to shortcut path as second) is CRUCIAL there
            next->forward(d_oCopyTensor, batchSize);
            nextShortcut->forward(d_inputTensor, batchSize);
        }
        // Case for training without skip
        else if (retainRate >= 1.0f || distribution(randomGenerator) <= static_cast<double>(retainRate)) {
            // Copy out the input tensor into 2 tensor for the splitting
            checkCuda(cudaMemcpy(d_oCopyTensor, d_inputTensor, fullSize * sizeof(float), cudaMemcpyDeviceToDevice));
            // Lend the both tensors to the next layers
            // The correct order (forward to shortcut path as second) is CRUCIAL there
            next->forward(d_oCopyTensor, batchSize);
            nextShortcut->forward(d_inputTensor, batchSize);
        }
        // Case for training with skip
        else {
            // Pass nullptr as tensor to the next merge layer (signaling skip)
            // The correct order (forward to shortcut path as second) is CRUCIAL there
            nextMerge->forward(nullptr, batchSize);
            // Lend the input tensor to next shortcut path layer
            nextShortcut->forward(d_inputTensor, batchSize);
        }
    }


    void backward(float* d_gradientTensor) override {
        // Reclaim the incoming gradient tensor from full path as owned copy tensor
        if (!partialBackward) {
            partialBackward = true;
            // Reclaim the incoming tensor or keep the current one based on SD skip
            if (d_gradientTensor == nullptr) stochasticSkip = true;
            else d_oCopyTensor = d_gradientTensor;
            return;
        }
        // Conditionally add the gradient tensor from full path into the shortcut path's tensor
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
        // Complete memory the trade by giving back the tensor with gradient data
        prev->backward(d_gradientTensor);
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
    Layer* prevShortcut;
    SplitLayer* prevSplit;
    float* d_bPrevTensor;

public:
    AddMergeLayer(const TensorSize inputSize): 
            Layer(inputSize, inputSize),
            partialForward(false),
            stochasticSkip(false),
            prevShortcut(nullptr),
            prevSplit(nullptr),
            d_bPrevTensor(nullptr) {}


    // Connect to the previous layers with inverted approach instead to avoid ambiguity
    virtual void _setPrev(Layer* prevLayer) {}


    // Connect the full incoming path
    void setPrev(Layer* prevLayer) {
        prevLayer->_setNext(this);
        prev = prevLayer;
    }


    // Connect the shortcut incoming path
    void setPrevShortcut(Layer* prevShortcutLayer) {
        prevShortcutLayer->_setNext(this);
        prevShortcut = prevShortcutLayer;
    }


    // Connect the corresponding split layer to conditionally skip full path
    void setPrevSplit(SplitLayer* prevSplitLayer) {
        prevSplitLayer->_setNextMerge(this);
        prevSplit = prevSplitLayer;
    }


    void forward(float* d_inputTensor, const size_t batchSize) override {
        // Save the batch size for backwards pass
        currBatchSize = batchSize;
        // Handle forward pass comming from the full path
        if (!partialForward) {
            partialForward = true;
            // Borrow the input tensor and conditionally set the SD flag
            d_bPrevTensor = d_inputTensor;
            stochasticSkip = (d_inputTensor == nullptr);
            return;
        }
        // Handle forward pass comming from the shortcut path
        partialForward = false;
        // Conditionally skip the addition and forward just the tensor from shortcut path
        if (!stochasticSkip) {
            // Add the input tensors into the full path's input tensor
            const size_t fullSize = batchSize * inputSize.fullSize();
            elementwiseAddInplace<<<ceilDiv(fullSize, BLOCK_SIZE), BLOCK_SIZE>>>(
                d_inputTensor, d_bPrevTensor, fullSize
            );
            checkCudaLastError();
        }
        // Lend the borrowed shortcut path's tensor with activation data to next layer
        next->forward(d_inputTensor, batchSize);
    }


    void backward(float* d_gradientTensor) override {
        // Conditionally skip backprob through the full path
        if (!stochasticSkip) {
            const size_t copySize = currBatchSize * inputSize.fullSize() * sizeof(float);
            checkCuda(cudaMemcpy(d_bPrevTensor, d_gradientTensor, copySize, cudaMemcpyDeviceToDevice));
            // Give back the full paths's tensor with gradient data
            prev->backward(d_bPrevTensor);
        }
        // Or signal the skip for the split layer
        else prevSplit->backward(nullptr);
        // Reset the partial forward and stochastic skip flags
        partialForward = false;
        stochasticSkip = false;
        // Complete the memory trade by giving back the shortcut path's tensor with gradient data
        // The correct order (backward to shortcut path as second) is CRUCIAL there
        prevShortcut->backward(d_gradientTensor);
    }

};



class MulMergeLayer : public Layer {
private:
    bool partialForward;
    bool stochasticSkip;
    Layer* prevShortcut;
    SplitLayer* prevSplit;
    float* d_oOutTensor;
    float* d_bPrevTensor;
    float* d_bPrevShortcutTensor;

public:
    MulMergeLayer(const TensorSize inputSize): 
            Layer(inputSize, inputSize),
            partialForward(false),
            stochasticSkip(false),
            prevShortcut(nullptr),
            prevSplit(nullptr),
            d_oOutTensor(nullptr),
            d_bPrevTensor(nullptr),
            d_bPrevShortcutTensor(nullptr) {}


    // Connect to the previous layers with inverted approach instead to avoid ambiguity
    virtual void _setPrev(Layer* prevLayer) {}


    // Connect the full incoming path
    void setPrev(Layer* prevLayer) {
        prevLayer->_setNext(this);
        prev = prevLayer;
    }


    // Connect the shortcut incoming path
    void setPrevShortcut(Layer* prevShortcutLayer) {
        prevShortcutLayer->_setNext(this);
        prevShortcut = prevShortcutLayer;
    }


    // Connect the corresponding split layer to conditionally skip the full path
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
        // Handle forward pass comming from the full path
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
        // Handle forward pass comming from the shortcut path
        partialForward = false;
        // Borrow the input tensor as the shortcut path's tensor
        d_bPrevShortcutTensor = d_inputTensor;
        // Case for inference with inplace multiplication into shortcut's tensor
        if (!backOn) {
            elementwiseMulInplace<<<ceilDiv(fullSize, BLOCK_SIZE), BLOCK_SIZE>>>(
                d_bPrevShortcutTensor, d_bPrevTensor, fullSize
            );
            checkCudaLastError();
            // Lend the borrowed shortcut's input tensor with activation data to next layer
            next->forward(d_bPrevShortcutTensor, batchSize);
        }
        // Case for training without skip with multiplication into output tensor
        else if (!stochasticSkip) {
            elementwiseMul<<<ceilDiv(fullSize, BLOCK_SIZE), BLOCK_SIZE>>>(
                d_oOutTensor, d_bPrevTensor, d_bPrevShortcutTensor, fullSize
            );
            checkCudaLastError();
            // Lend the output tensor with activation data to next layer
            next->forward(d_oOutTensor, batchSize);
        }
        // Case for training without skip
        else {
            // Lend the borrowed shortcut's input tensor with it's data to next layer
            next->forward(d_bPrevShortcutTensor, batchSize);
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
                d_bPrevTensor, d_bPrevShortcutTensor, d_oOutTensor, fullSize
            );
            checkCudaLastError();
            // Complete the memory trade by giving back the tensors with gradient data
            // The correct order (backward to shortcut path as second) is CRUCIAL there
            prev->backward(d_bPrevTensor);
            prevShortcut->backward(d_bPrevShortcutTensor);
        }
        // Case with skip
        else {
            // Signal the skip for the split layer
            prevSplit->backward(nullptr);
            // The correct order (backward to shortcut path as second) is CRUCIAL there
            // Complete the memory trade by giving back the shortcut path's tensor with gradient data
            //  since the borrowed shortcut path's tensor was given to this layer instead of output tensor
            prevShortcut->backward(d_gradientTensor);
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
    float dropoutRate;
    size_t dropoutSeed;
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
            const size_t seed,
            const cudnnHandle_t handle): 
                Layer(inputSize, inputSize),
                dropoutRate(rate),
                dropoutSeed(seed),
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
        if (!backOn && trainOn) {
            backOn = true;
            // Recreate the dropout states the training
            checkCuda(cudaMalloc(&d_oCudnnDropoutStates, cudnnDropoutStatesSize));
            checkCudnn(cudnnSetDropoutDescriptor(
                cudnnDropoutDesc, cudnnHandle, dropoutRate,
                d_oCudnnDropoutStates, cudnnDropoutStatesSize, dropoutSeed
            ));
        }
        else if (backOn && !trainOn) {
            backOn = false;
            // Free the rosources needed only for training
            checkCuda(cudaFree(d_oCudnnDropoutStates));
            checkCuda(cudaFree(d_oCudnnReserveSpace));
            d_oCudnnDropoutStates = nullptr;
            d_oCudnnReserveSpace = nullptr;
        }
        // Reset the batch sizes to force buffer reallocations in forward pass
        if (backOn != trainOn) {
            cudnnReserveSpaceSize = 0;
            currBatchSize = 0;
            currActualBatchSize = 0;
        }
    };


    void forward(float* d_inputTensor, const size_t batchSize) override {
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
        checkCudnn(cudnnDropoutBackward(
            cudnnHandle, cudnnDropoutDesc,
            cudnnTensorDesc, d_gradientTensor,
            cudnnTensorDesc, d_gradientTensor,
            d_oCudnnReserveSpace, cudnnReserveSpaceSize
        ));
        // Complete memory the trade by giving back the recieved tensor with gradient data
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



template <typename ACTIVATION>
class ActivationLayer : public Layer {
private:
    float* d_oSavedTensor;

public:
    ActivationLayer(const TensorSize inputSize): 
            Layer(inputSize, inputSize),
            d_oSavedTensor(nullptr) {}


    void toggleTrain(const bool trainOn) override {
        if (!backOn && trainOn) backOn = true;
        else if (backOn && !trainOn) {
            backOn = false;
            // Free the buffer only needed during training
            checkCuda(cudaFree(d_oSavedTensor));
            d_oSavedTensor = nullptr;
        }
        // Reset the batch sizes to force buffer reallocations in forward pass
        if (backOn != trainOn) {
            currBatchSize = 0;
            currActualBatchSize = 0;
        }
    };


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


    void forward(float* d_inputTensor, const size_t batchSize) override {
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
        const size_t fullSize = currBatchSize * inputSize.fullSize();
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


    ~AvgPoolingFlattenLayer() {
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


    void forward(float* d_inputTensor, const size_t batchSize) override {
        const size_t fullSize = batchSize * outputSize.fullSize();
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
        if (!backOn && trainOn) {
            backOn = true;
            // Allocate resources independant on batch size
            checkCuda(cudaMalloc(&d_oScaleGrads, outputSize.C * sizeof(float)));
            checkCuda(cudaMalloc(&d_oShiftGrads, outputSize.C * sizeof(float)));
            checkCuda(cudaMalloc(&d_oBatchMean, outputSize.C * sizeof(float)));
            checkCuda(cudaMalloc(&d_oBatchInvVariance, outputSize.C * sizeof(float)));
        }
        else if (backOn && !trainOn) {
            backOn = false;
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
        if (backOn != trainOn) {
            currBatchSize = 0;
            currActualBatchSize = 0;
        }
    };


    void initWeights() {
        // Initialize the scale, shift, running mean and variance to default values
        initValues<<<ceilDiv(outputSize.C, BLOCK_SIZE), BLOCK_SIZE>>>(d_oScale, 1.0f, outputSize.C);
        checkCudaLastError();
        initValues<<<ceilDiv(outputSize.C, BLOCK_SIZE), BLOCK_SIZE>>>(d_oShift, 0.0f, outputSize.C);
        checkCudaLastError();
        initValues<<<ceilDiv(outputSize.C, BLOCK_SIZE), BLOCK_SIZE>>>(d_oRunningMean, 0.0f, outputSize.C);
        checkCudaLastError();
        initValues<<<ceilDiv(outputSize.C, BLOCK_SIZE), BLOCK_SIZE>>>(d_oRunningVar, 1.0f, outputSize.C);
        checkCudaLastError();
    }


    void registerWeights(Optimizer& optimizer) {
        if (backOn) {
            // Register the scale and shift perameter tensors and their corresponding gradient tensors
            optimizer.registerLayer(optimizerAlgorithm, d_oScale, d_oScaleGrads, outputSize.C);
            optimizer.registerLayer(optimizerAlgorithm, d_oShift, d_oShiftGrads, outputSize.C);
        }
    }


    void forward(float* d_inputTensor, const size_t batchSize) override {
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
                const size_t fullSizeBytes = batchSize * outputSize.fullSize() * sizeof(float);
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
        // Compute the parameter gradients and input gradients
        const float alpha = 1.0f;
        const float beta = 0.0f;
        checkCudnn(cudnnBatchNormalizationBackward(
            cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
            &alpha, &beta, &alpha, &beta,
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


    ~BatchNormLayer() {
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


    void forward(float* d_inputTensor, const size_t batchSize) override {
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

    
    // Does not use the backward function, handles host tensor instead
    void backward(float* d_gradientTensor) override {}


    // Fills the host buffer with the calculated probabilities
    // Only valid before starting backprop
    void getHostProbs(float* probTensor) {
        const size_t copySizeBytes = currBatchSize * inputSize.C * sizeof(float);
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
        const size_t fullSize = currBatchSize * inputSize.C;
        crossEntropyLossGradInplace<<<ceilDiv(fullSize, BLOCK_SIZE), BLOCK_SIZE>>>(
            d_bPrevTensor, d_oLabelValues, currBatchSize, inputSize.C
        );
        checkCudaLastError();
        // Initiate the packprop by returning the borrowed tensor with loss greadient data
        prev->backward(d_bPrevTensor);
    }


    ~SoftmaxLossLayer() {
        checkCuda(cudaFree(d_oLabelValues));
        checkCuda(cudaFree(d_oLossValues));
        checkCudnn(cudnnDestroyTensorDescriptor(cudnnTensorDesc));
    }

};
