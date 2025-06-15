#pragma once
#include <random>
#include <vector>
#include <memory>
#include <numeric>
#include "../layers/Layer.cuh"
#include "../layers/Convolution.cuh"
#include "../network/Optimizer.cuh"
#include "../utils/Math.cuh"



constexpr size_t FILTER_SIZE = 3;   // template later 
constexpr size_t STRIDE = 1;        // template later
class MBConvBlock {
private:
    std::vector<std::unique_ptr<Layer>> allLayers;
    std::vector<LearnableLayer*> learnableLayers;
    Layer *firstLayer, *lastLayer;
    std::unique_ptr<std::mt19937> randomGenerator;

public:
    MBConvBlock(
            const TensorSize inputSize,
            const size_t outChannels,
            const size_t expandRatio,
            const size_t inverseSERatio,
            const float stochasticDepthRate,
            const size_t seed) {
        // Initialize MBConvBlock local random generator
        randomGenerator = std::make_unique<std::mt19937>(seed);
        
        
        












    
    }


    Layer* getFirstLayer() { return firstLayer; }
    void connect(Layer* nextLayer) { lastLayer->connect(nextLayer); }


    void toggleTrain(const bool trainOn) {
        for (std::unique_ptr<Layer>& layer : allLayers) {
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

};
