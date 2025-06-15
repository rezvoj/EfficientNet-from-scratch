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
        const TensorSize inputSize = {3, 224, 224};
        // EfficientNet constants
        constexpr float epsilon = 1e-3;
        constexpr float expAvgFactor = 0.01;
        // Host to device input layer
        startLayer = new InputLayer(inputSize);
        allLayers.push_back(startLayer);
        // Stage 1 initial stem convolution layers
        auto stemConv = new ConvolutionLayer(inputSize, 32, 3, 2, true, Optimizer::ADAM_W, cudnnHandle);
        startLayer->connect(stemConv);
        allLayers.push_back(stemConv);
        TensorSize currSize = {32, 112, 112};
        auto stemBN = new BatchNormLayer(currSize, expAvgFactor, epsilon, Optimizer::ADAM, cudnnHandle);
        stemConv->connect(stemBN);
        allLayers.push_back(stemBN);
        auto stemSiLU = new ActivationLayer<SiLU>(currSize);
        stemBN->connect(stemSiLU);
        allLayers.push_back(stemSiLU);
        // Stage 2 MBConv block 1
        auto mbconv1 = new MBConvBlock<3, 1>(currSize, 16, 1, 0,
            const float SDRetainRate,
            const float epsilon,
            const float expAvgFactor,
            const uint seed,
            const cudnnHandle_t cudnnHandle,
            const cublasHandle_t cublasHandle);




        // Now what ? 

        Block Index	| Stage | Input Channels | Output Channels | Kernel	| Stride | Expand Ratio | SE Ratio | Stochastic Depth Rate
        0	2	32	16	3x3	1	1	0.0	    0.0
                                        
        1	3	16	24	3x3	2	6	0.25	0.0125
        2	3	24	24	3x3	1	6	0.25	0.025
                                        
        3	4	24	40	5x5	2	6	0.25	0.0375
        4	4	40	40	5x5	1	6	0.25	0.05
                                        
        5	5	40	80	3x3	2	6	0.25	0.0625
        6	5	80	80	3x3	1	6	0.25	0.075
        7	5	80	80	3x3	1	6	0.25	0.0875
                     
        
        8	6	80	112	5x5	1	6	0.25	0.1
        9	6	112	112	5x5	1	6	0.25	0.1125
        10	6	112	112	5x5	1	6	0.25	0.125
        
        
        11	7	112	192	5x5	2	6	0.25	0.1375
        12	7	192	192	5x5	1	6	0.25	0.15
        13	7	192	192	5x5	1	6	0.25	0.1625
        14	7	192	192	5x5	1	6	0.25	0.175
        
        
        15	8	192	320	3x3	1	6	0.25	0.1875











        // Device to host output cross entropy softmax layer
        endLayer = new SoftmaxLossLayer(/*placeholder*/ TensorSize{1, 1, 1}, epsilon, cudnnHandle);
        allLayers.push_back(endLayer);
    }


    TensorSize getInputSize() const {
        return TensorSize{3, 224, 224};
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


    void predict(float* probabilities, const float* inputTensor, const uint batchSize) {
        if (trainOn) return;
        startLayer->startForward(inputTensor, batchSize);
        endLayer->getHostProbs(probabilities);
    }


    void predict(
            float* batchLoss,
            const float* inputTensor,
            const uint* trueLabels,
            const uint batchSize) {
        if (trainOn) return;
        startLayer->startForward(inputTensor, batchSize);
        endLayer->getHostBatchLoss(batchLoss, trueLabels);
    }


    void forwardBackward(
            float* batchLoss,
            const float* inputTensor,
            const uint* trueLabels,
            const uint batchSize) {
        if (!trainOn) return;
        startLayer->startForward(inputTensor, batchSize);
        endLayer->getHostBatchLoss(batchLoss, trueLabels);
        endLayer->startBackward(nullptr);
    }


    ~EfficientNetB0() {
        delete randomGenerator;
        for (Layer* layer : allLayers) delete layer;
        for (IMBConvBlock* block : mbconvBlocks) delete block;
    }

};
