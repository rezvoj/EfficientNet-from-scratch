#pragma once
#include "../network/Optimizer.cuh"

constexpr uint BLOCK_SIZE = 256;
constexpr uint BLOCK_X_SIZE = 16;
constexpr uint BLOCK_Y_SIZE = 16;



struct TensorSize {
    uint C = 1;
    uint H = 1;
    uint W = 1;
    
    uint fullSize() const {
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



class Layer {
protected:
    bool backOn;
    TensorSize inputSize;
    TensorSize outputSize;
    uint currBatchSize;
    uint currActualBatchSize;
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
    virtual Layer* connect(Layer* nextLayer) {
        next = nextLayer;
        nextLayer->_setPrev(this);
        return this;
    }
    
    // Changes the layer from training to inference mode and vice versa
    // Allocating and freeing buffers which are / are not needed
    virtual void toggleTrain(const bool trainOn) {
        if (backOn == trainOn) return;
        backOn = trainOn;
        // Reset the batch sizes to force buffer reallocations
        currBatchSize = 0;
        currActualBatchSize = 0;
    }
    
    // Performs forward pass calculation for the current layer
    // Potentially saves the borrowed tensor to non owning pointer
    // Borrows it's output tensor with the activation data to the next layer
    virtual void forward(float* d_inputTensor, const uint batchSize) = 0;
    
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
    virtual void initWeights(const uint seed) = 0;

    // Registers the layer's weight and weight gradient tensors
    // The registered tensors CAN'T be exchanged with other layers in memory trade
    // Doesn't do anything if layer is in inference mode
    virtual void registerWeights(Optimizer& optimizer) = 0;

    // Called after switching train to inference and back to train mode during training loop
    // Doesn't do anything if layer is still in inference mode
    virtual void reRegisterGrads(Optimizer& optimizer) = 0;
};
