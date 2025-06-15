#pragma once
#include <random>
#include <vector>
#include <algorithm>
#include "../layers/Layer.cuh"
#include "../layers/Activation.cuh"
#include "../layers/Convolution.cuh"
#include "../layers/FirstLast.cuh"
#include "../layers/Linear.cuh"
#include "../layers/Regularization.cuh"
#include "../layers/Resizing.cuh"
#include "../utils/Math.cuh"
#include "Optimizer.cuh"
#include "MBConv.cuh"



class EfficientNetB0 {
private:
    bool trainOn;
    std::vector<IMBConvBlock*> mbconvBlocks;
    std::vector<Layer*> allLayers;
    InputLayer* startLayer;
    SoftmaxLossLayer* endLayer;
    std::mt19937* randomGenerator;
    cudnnHandle_t cudnnHandle;
    cublasHandle_t cublasHandle;

public:
    EfficientNetB0(const uint classes, const uint seed) : trainOn(false) {
        randomGenerator = new std::mt19937(seed);














    }


    void predict() {
        if (trainOn) return;
        





    }


    void forwardBackward() {
        if (!trainOn) return;



    }


    void toggleTrain(const bool trainOn) {
        this->trainOn = trainOn;
        for (Layer* layer : allLayers) {
            layer->toggleTrain(trainOn);
        }
        for (IMBConvBlock* block : mbconvBlocks) {
            block->toggleTrain(trainOn);
        }
    }
    

    void initWeights() {
        for (Layer* layer : allLayers) {
            if (LearnableLayer* learnable = dynamic_cast<LearnableLayer*>(layer)) {
                learnable->initWeights((*randomGenerator)());
            }
        }
        for (IMBConvBlock* block : mbconvBlocks) {
            block->initWeights();
        }
    }


    void registerWeights(Optimizer& optimizer) {
        for (Layer* layer : allLayers) {
            if (LearnableLayer* learnable = dynamic_cast<LearnableLayer*>(layer)) {
                learnable->registerWeights(optimizer);
            }
        }
        for (IMBConvBlock* block : mbconvBlocks) {
            block->registerWeights(optimizer);
        }
    }


    void reRegisterGrads(Optimizer& optimizer) {
        for (Layer* layer : allLayers) {
            if (LearnableLayer* learnable = dynamic_cast<LearnableLayer*>(layer)) {
                learnable->reRegisterGrads(optimizer);
            }
        }
        for (IMBConvBlock* block : mbconvBlocks) {
            block->reRegisterGrads(optimizer);
        }
    }


    ~EfficientNetB0() {
        delete randomGenerator;
        for (Layer* layer : allLayers) delete layer;
        for (IMBConvBlock* block : mbconvBlocks) delete block;
    }

};
