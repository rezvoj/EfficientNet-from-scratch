#pragma once
#include <exception>
#include <random>
#include "Layer.cuh"
#include "../kernels/Elementwise.cuh"
#include "../utils/Exceptions.cuh"
#include "../utils/Math.cuh"

constexpr uint BLOCK_SIZE = 256;



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
    SplitLayer(
            const TensorSize inputSize,
            const float retainRate,
            const uint seed):
                Layer(inputSize, inputSize),
                partialBackward(false),
                stochasticSkip(false),
                retainRate(retainRate),
                nextShortcut(nullptr),
                nextMerge(nullptr),
                d_oCopyTensor(nullptr),
                randomGenerator(nullptr) {
        if (retainRate < 1.0f) {
            randomGenerator = new std::mt19937(seed);
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


    void forward(float* d_inputTensor, const uint batchSize) override {
        const uint fullSize = batchSize * inputSize.fullSize();
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
            const uint fullSize = currBatchSize * inputSize.fullSize();
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
            delete randomGenerator;
            checkCuda(cudaFree(d_oCopyTensor));
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


    // Normally connected layer is the shortcut path layer
    void _setPrev(Layer* prevShortcutLayer) override {
        prevShortcut = prevShortcutLayer;
    }


    // Connect to the full layers with inverted approach instead to avoid ambiguity
    void setPrevFull(Layer* prevLayer) {
        prevLayer->_setNext(this);
        prev = prevLayer;
    }


    // Connect the corresponding split layer with inverted approach 
    //  to conditionally skip full path
    void setPrevSplit(SplitLayer* prevSplitLayer) {
        prevSplitLayer->_setNextMerge(this);
        prevSplit = prevSplitLayer;
    }


    void forward(float* d_inputTensor, const uint batchSize) override {
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
            const uint fullSize = batchSize * inputSize.fullSize();
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
            const uint copySize = currBatchSize * inputSize.fullSize() * sizeof(float);
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

    
    // Normally connected layer is the shortcut path layer
    void _setPrev(Layer* prevShortcutLayer) override {
        prevShortcut = prevShortcutLayer;
    }


    // Connect to the full layers with inverted approach instead to avoid ambiguity
    void setPrevFull(Layer* prevLayer) {
        prevLayer->_setNext(this);
        prev = prevLayer;
    }


    // Connect the corresponding split layer with inverted approach 
    //  to conditionally skip full path
    void setPrevSplit(SplitLayer* prevSplitLayer) {
        prevSplitLayer->_setNextMerge(this);
        prevSplit = prevSplitLayer;
    }


    void toggleTrain(const bool trainOn) override {
        if (backOn == trainOn) return;
        backOn = trainOn;
        if (!trainOn) {
            // Free the unnecessary memory for inference
            checkCuda(cudaFree(d_oOutTensor));
            d_oOutTensor = nullptr;
        }
        // Reset the batch sizes to force buffer reallocations
        currBatchSize = 0;
        currActualBatchSize = 0;
    }


    void forward(float* d_inputTensor, const uint batchSize) override {
        const uint fullSize = batchSize * inputSize.fullSize();
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
            const uint fullSize = currBatchSize * inputSize.fullSize();
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
