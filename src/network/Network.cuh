#pragma once
#include <random>
#include <vector>
#include <algorithm>
#include <cudnn_v9.h>
#include <cublas_v2.h>
#include "../layers/Layer.cuh"
#include "../layers/Activation.cuh"
#include "../layers/Convolution.cuh"
#include "../layers/FirstLast.cuh"
#include "../layers/Linear.cuh"
#include "../layers/Regularization.cuh"
#include "../layers/Resizing.cuh"
#include "../utils/Exceptions.cuh"
#include "../utils/Math.cuh"
#include "Optimizer.cuh"
#include "MBConv.cuh"



class EfficientNetB0 {
private:
    bool trainOn;
    std::vector<IMBConvBlock*> mbconvBlocks;
    std::vector<Layer*> otherLayers;
    InputLayer* startLayer;
    SoftmaxLossLayer* endLayer;
    std::mt19937* randomGenerator;
    cudnnHandle_t cudnnHandle;
    cublasHandle_t cublasHandle;

private:
    template <uint FILTER_SIZE, uint STRIDE>
    IMBConvBlock* appendFollowingMBConv(
            const TensorSize inputSize,
            const uint outChannels,
            const float SDRetainRate) {
        constexpr float epsilon = 1e-3;
        constexpr float expAvgFactor = 0.01;
        auto mbconv = new MBConvBlock<FILTER_SIZE, STRIDE>(
            inputSize, outChannels, 6, 4, SDRetainRate, epsilon, expAvgFactor, 
            (*randomGenerator)(), cudnnHandle, cublasHandle
        );
        mbconvBlocks.back()->connect(mbconv->getFirstLayer());
        mbconvBlocks.push_back(mbconv);
    }

public:
    EfficientNetB0(const uint numClasses, const uint seed) : trainOn(false) {
        // EfficientNet constants
        constexpr TensorSize inputSize = {3, 224, 224};
        constexpr float epsilon = 1e-3;
        constexpr float expAvgFactor = 0.01;
        // Initialize resources
        checkCudnn(cudnnCreate(&cudnnHandle));
        checkCublas(cublasCreate_v2(&cublasHandle));
        randomGenerator = new std::mt19937(seed);
        // Host to device input layer
        startLayer = new InputLayer(inputSize);
        otherLayers.push_back(startLayer);
        // Stage 1 initial stem convolution layers
        auto stemConv = new ConvolutionLayer(
            inputSize, 32, 3, 2, true,
            Optimizer::ADAM_W, cudnnHandle
        );
        startLayer->connect(stemConv);
        otherLayers.push_back(stemConv);
        const TensorSize afterStemSize = {32, 112, 112};
        auto stemBN = new BatchNormLayer(
            afterStemSize, expAvgFactor, epsilon, 
            Optimizer::ADAM, cudnnHandle
        );
        stemConv->connect(stemBN);
        otherLayers.push_back(stemBN);
        auto stemSiLU = new ActivationLayer<SiLU>(afterStemSize);
        stemBN->connect(stemSiLU);
        otherLayers.push_back(stemSiLU);
        // Stage 2 initial MBConv block 1
        auto mbconv1 = new MBConvBlock<3, 1>(
            afterStemSize, 16, 1, 0, 1.0f, epsilon, expAvgFactor, 
            (*randomGenerator)(), cudnnHandle, cublasHandle
        );
        stemSiLU->connect(mbconv1->getFirstLayer());
        mbconvBlocks.push_back(mbconv1);
        // Stage 3 MBConv blocks 2-3
        appendFollowingMBConv<3, 2>({16, 112, 112}, 24, 0.9875f);
        appendFollowingMBConv<3, 1>({24, 56, 56}, 24, 0.975f);
        // Stage 4 MBConv blocks 4-5
        appendFollowingMBConv<5, 2>({24, 56, 56}, 40, 0.9625f);
        appendFollowingMBConv<5, 1>({40, 28, 28}, 40, 0.95f);
        // Stage 5 MBConv blocks 6-8
        appendFollowingMBConv<3, 2>({40, 28, 28}, 80, 0.9375f);
        appendFollowingMBConv<3, 1>({80, 14, 14}, 80, 0.925f);
        appendFollowingMBConv<3, 1>({80, 14, 14}, 80, 0.9125f);
        // Stage 6 MBConv blocks 9-11
        appendFollowingMBConv<5, 1>({80, 14, 14}, 112, 0.9f);
        appendFollowingMBConv<5, 1>({112, 14, 14}, 112, 0.8875f);
        appendFollowingMBConv<5, 1>({112, 14, 14}, 112, 0.875f);
        // Stage 7 MBConv blocks 12-15
        appendFollowingMBConv<5, 2>({112, 14, 14}, 192, 0.8625f);
        appendFollowingMBConv<5, 1>({192, 7, 7}, 192, 0.85f);
        appendFollowingMBConv<5, 1>({192, 7, 7}, 192, 0.8375f);
        appendFollowingMBConv<5, 1>({192, 7, 7}, 192, 0.825f);
        // Stage 8 final MBConv block 16
        auto mbconv16 = appendFollowingMBConv<3, 1>({192, 7, 7}, 320, 0.8125f);
        // Stage 9 classification head spatial part
        const uint headChannels = 1280;
        auto headConv = new ConvolutionLayer(
            {320, 7, 7}, headChannels, 1, 1, false,
            Optimizer::ADAM_W, cudnnHandle);
        mbconv16->connect(headConv);
        otherLayers.push_back(headConv);
        const TensorSize headConvOutSize = {headChannels, 7, 7};
        auto headBN = new BatchNormLayer(
            headConvOutSize, expAvgFactor, epsilon,
            Optimizer::ADAM, cudnnHandle
        );
        headConv->connect(headBN);
        otherLayers.push_back(headBN);
        auto headSiLU = new ActivationLayer<SiLU>(headConvOutSize);
        headBN->connect(headSiLU);
        otherLayers.push_back(headSiLU);
        // Stage 10 classification head flat part
        auto globalPool = new AvgPoolingFlattenLayer(headConvOutSize, cudnnHandle);
        headSiLU->connect(globalPool);
        otherLayers.push_back(globalPool);
        const TensorSize pooledOutSize = {headChannels, 1, 1};
        auto dropout = new DropoutLayer(pooledOutSize, 0.2f, (*randomGenerator)(), false, cudnnHandle);
        globalPool->connect(dropout);
        otherLayers.push_back(dropout);
        auto classifier = new LinearLayer(
            headChannels, numClasses, false,
            Optimizer::ADAM_W, cublasHandle
        );
        dropout->connect(classifier);
        otherLayers.push_back(classifier);
        // Device to host output cross entropy softmax layer
        endLayer = new SoftmaxLossLayer(TensorSize{numClasses, 1, 1}, epsilon, cudnnHandle);
        classifier->connect(endLayer);
        otherLayers.push_back(endLayer);
    }


    TensorSize getInputSize() const {
        return TensorSize{3, 224, 224};
    }


    void toggleTrain(const bool trainOn) {
        this->trainOn = trainOn;
        for (Layer* layer : otherLayers) {
            layer->toggleTrain(trainOn);
        }
        for (IMBConvBlock* block : mbconvBlocks) {
            block->toggleTrain(trainOn);
        }
    }
    

    void initWeights() {
        for (Layer* layer : otherLayers) {
            if (LearnableLayer* learnable = dynamic_cast<LearnableLayer*>(layer)) {
                learnable->initWeights((*randomGenerator)());
            }
        }
        for (IMBConvBlock* block : mbconvBlocks) {
            block->initWeights();
        }
    }


    void registerWeights(Optimizer& optimizer) {
        for (Layer* layer : otherLayers) {
            if (LearnableLayer* learnable = dynamic_cast<LearnableLayer*>(layer)) {
                learnable->registerWeights(optimizer);
            }
        }
        for (IMBConvBlock* block : mbconvBlocks) {
            block->registerWeights(optimizer);
        }
    }


    void reRegisterGrads(Optimizer& optimizer) {
        for (Layer* layer : otherLayers) {
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
        checkCudnn(cudnnDestroy(cudnnHandle));
        checkCublas(cublasDestroy_v2(cublasHandle));
        delete randomGenerator;
        for (Layer* layer : otherLayers) delete layer;
        for (IMBConvBlock* block : mbconvBlocks) delete block;
    }

};
