#pragma once
#include <random>
#include <vector>
#include <algorithm>
#include "../layers/Layer.cuh"
#include "../layers/Activation.cuh"
#include "../layers/Convolution.cuh"
#include "../layers/Linear.cuh"
#include "../layers/Regularization.cuh"
#include "../layers/Resizing.cuh"
#include "../layers/SplitMerge.cuh"
#include "../network/Optimizer.cuh"
#include "../utils/Math.cuh"



template <uint FILTER_SIZE, uint STRIDE>
class MBConvBlock {
private:
    std::vector<Layer*> allLayers;
    std::vector<LearnableLayer*> learnableLayers;
    Layer *firstLayer, *lastLayer;
    std::mt19937* randomGenerator;

public:
    MBConvBlock(
            const TensorSize inputSize,
            const uint outChannels,
            const uint expandRatio,
            const uint inverseSERatio,
            const float SDRetainRate,
            const float epsilon,
            const float expAvgFactor,
            const uint seed,
            const cudnnHandle_t cudnnHandle,
            const cublasHandle_t cublasHandle) {
        randomGenerator = new std::mt19937(seed);
        const TensorSize expandConvInputSize = {inputSize.C * expandRatio, inputSize.H, inputSize.W};
        // Depthwise convolution layers
        auto depthwiseConv = new DepthwiseConvolutionLayer<FILTER_SIZE, STRIDE>(expandConvInputSize, false, Optimizer::ADAM_W);
        const TensorSize depthwiseOutputSize = {inputSize.C * expandRatio, ceilDiv(inputSize.H, STRIDE), ceilDiv(inputSize.W, STRIDE)};
        auto depthwiseBN = new BatchNormLayer(depthwiseOutputSize, expAvgFactor, epsilon, Optimizer::ADAM, cudnnHandle);
        auto depthwiseSiLU = new ActivationLayer<SiLU>(depthwiseOutputSize);
        // Save and connect the depthwise convolution layers
        learnableLayers.insert(learnableLayers.end(), {depthwiseConv, depthwiseBN});
        allLayers.insert(allLayers.end(), {depthwiseConv, depthwiseBN, depthwiseSiLU});
        depthwiseConv->connect(depthwiseBN->connect(depthwiseSiLU));
        // Squeeze-excitation block layers
        auto seSplit = new SplitLayer(depthwiseOutputSize, 1.0f, (*randomGenerator)());
        auto seFlatten = new AvgPoolingFlattenLayer(depthwiseOutputSize, cudnnHandle);
        const uint seSqueezedChannels = std::max(1UL, inputSize.C / inverseSERatio);
        auto seBottleneck = new LinearLayer(depthwiseOutputSize.C, seSqueezedChannels, false, Optimizer::ADAM_W, cublasHandle);
        auto seSiLU = new ActivationLayer<SiLU>(TensorSize{seSqueezedChannels, 1, 1});
        auto seLinearOut = new LinearLayer(seSqueezedChannels, depthwiseOutputSize.C, false, Optimizer::ADAM_W, cublasHandle);
        auto seSigmoid = new ActivationLayer<Sigmoid>(TensorSize{depthwiseOutputSize.C, 1, 1});
        auto seExpansion = new ExpansionLayer(depthwiseOutputSize, cudnnHandle);
        auto seMulMerge = new MulMergeLayer(depthwiseOutputSize);
        // Connect and save the squeeze-excitation block layers
        learnableLayers.insert(learnableLayers.end(), {seBottleneck, seLinearOut});
        allLayers.insert(allLayers.end(), {seSplit, seFlatten, seBottleneck, seSiLU, seLinearOut, seSigmoid, seExpansion, seMulMerge});
        depthwiseSiLU->connect(seSplit->connect(seFlatten->connect(seBottleneck)));
        seBottleneck->connect(seSiLU->connect(seLinearOut->connect(seSigmoid->connect(seExpansion))));
        seMulMerge->setPrevFull(seExpansion);
        seMulMerge->setPrevSplit(seSplit);
        seSplit->connectShortcut(seMulMerge);
        // Shrinking convolution layers
        auto shrink = new ConvolutionLayer(depthwiseOutputSize, outChannels, 1, 1, false, Optimizer::ADAM_W, cudnnHandle);
        const TensorSize shrinkConvOutputSize = {outChannels, depthwiseOutputSize.H, depthwiseOutputSize.W};
        auto shrinkBN = new BatchNormLayer(shrinkConvOutputSize, expAvgFactor, epsilon, Optimizer::ADAM, cudnnHandle);
        // Connect and save the shrinking convolution layers
        learnableLayers.insert(learnableLayers.end(), {shrink, shrinkBN});
        allLayers.insert(allLayers.end(), {shrink, shrinkBN});
        seMulMerge->connect(shrink->connect(shrinkBN));
        // Set defualt first and last layers
        firstLayer = depthwiseConv;
        lastLayer = shrinkBN;
        // Conditionally add the expansion convolution layers
        if (expandRatio > 1) {
            auto expConv = new ConvolutionLayer(inputSize, inputSize.C * expandRatio, 1, 1, false, Optimizer::ADAM_W, cudnnHandle);
            auto expBN = new BatchNormLayer(expandConvInputSize, expAvgFactor, epsilon, Optimizer::ADAM, cudnnHandle);
            auto expSiLU = new ActivationLayer<SiLU>(expandConvInputSize);
            // Connect and save the expansion convolution layers
            learnableLayers.insert(learnableLayers.end(), {expConv, expBN});
            allLayers.insert(allLayers.end(), {expConv, expBN, expSiLU});
            expConv->connect(expBN->connect(expSiLU->connect(firstLayer)));
            firstLayer = expConv;
        }
        // Conditionally add the residual skip layers
        if (STRIDE == 1 && inputSize.C == outChannels) {
            auto resSplit = new SplitLayer(inputSize, SDRetainRate, (*randomGenerator)());
            auto resAddMerge = new AddMergeLayer(shrinkConvOutputSize);
            // Connect and save the residual skip layers
            allLayers.insert(allLayers.end(), {resSplit, resAddMerge});
            resSplit->connect(firstLayer);
            resAddMerge->setPrevFull(lastLayer);
            resAddMerge->setPrevSplit(resSplit);
            resSplit->connectShortcut(resAddMerge);
            firstLayer = resSplit;
            lastLayer = resAddMerge;
        }
    }


    Layer* getFirstLayer() { return firstLayer; }
    void connect(Layer* nextLayer) { lastLayer->connect(nextLayer); }


    void toggleTrain(const bool trainOn) {
        for (Layer* layer : allLayers) {
            layer->toggleTrain(trainOn);
        }
    }
    

    void initWeights() {
        for (LearnableLayer* layer : learnableLayers) { 
            layer->initWeights((*randomGenerator)());
        }
    }


    void registerWeights(Optimizer& optimizer) {
        for (LearnableLayer* layer : learnableLayers) { 
            layer->registerWeights(optimizer);
        }
    }


    void reRegisterGrads(Optimizer& optimizer) {
        for (LearnableLayer* layer : learnableLayers) {
            layer->reRegisterGrads(optimizer);
        }
    }

    ~MBConvBlock() {
        delete randomGenerator;
        for (Layer* layer : allLayers) {
            delete layer;
        }
    }

};
