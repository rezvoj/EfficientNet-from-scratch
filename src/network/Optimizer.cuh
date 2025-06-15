#pragma once
#include <vector>
#include "../kernels/Initialization.cuh"
#include "../kernels/Optimizer.cuh"
#include "../utils/Exceptions.cuh"
#include "../utils/Math.cuh"

constexpr uint BLOCK_SIZE = 256;



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



class Optimizer {
private:
    uint iteration;
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
        float* d_batchGradTensor;
        uint tensorFullSize;

        AdamWeightData(
                float* d_weightTensor,
                float* d_batchGradTensor,
                uint tensorFullSize) {
            this->d_weightTensor = d_weightTensor;
            this->d_batchGradTensor = d_batchGradTensor;
            this->tensorFullSize = tensorFullSize; 
            checkCuda(cudaMalloc(&d_oMMomentTensor, tensorFullSize * sizeof(float)));
            checkCuda(cudaMalloc(&d_oVMomentTensor, tensorFullSize * sizeof(float)));
            const dim3 gridSize(ceilDiv(tensorFullSize, BLOCK_SIZE));
            initValues<<<gridSize, BLOCK_SIZE>>>(d_oMMomentTensor, 0.0f, tensorFullSize);
            checkCudaLastError();
            initValues<<<gridSize, BLOCK_SIZE>>>(d_oVMomentTensor, 0.0f, tensorFullSize);
            checkCudaLastError();
        }        
    };

private:
    std::vector<AdamWeightData> adamParams;
    std::vector<AdamWeightData> adamWParams;

public:
    enum OptimAlgo { ADAM, ADAM_W };

private:
    static void zeroTensor(float* d_tensor, const uint tensorFullSize) {
        checkCuda(cudaMemset(d_tensor, 0, tensorFullSize * sizeof(float)));
    }

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
    // Zeroes the provided gradient accumulation tensor
    // The registered tensors are assumed to become
    //  invalid when layer switches to inference mode
    //  so the optimizer needs to be reset
    void registerLayer(
            const OptimAlgo algorithm,
            float* d_weightTensor,
            float* d_batchGradTensor,
            const uint tensorFullSize) {        
        switch (algorithm) {
            case ADAM:
                adamParams.push_back(AdamWeightData(d_weightTensor, d_batchGradTensor, tensorFullSize));
                zeroTensor(d_batchGradTensor, tensorFullSize);
                break;
            case ADAM_W: 
                adamWParams.push_back(AdamWeightData(d_weightTensor, d_batchGradTensor, tensorFullSize));
                zeroTensor(d_batchGradTensor, tensorFullSize);
                break;
        }
    }


    // Called after switching train to inference and back to train mode during training loop
    //  if the internal layer's gradient tensor was reallocated during the mode switching
    // Zeroes the provided gradient accumulation tensor
    void reRegisterLayerGrads(
            const OptimAlgo algorithm,
            float* d_keyWeightTensor,
            float* d_newBatchGradTensor) {
        switch (algorithm) {
            case ADAM:
                for (AdamWeightData& layerData : adamParams) {
                    if (layerData.d_weightTensor == d_keyWeightTensor) {
                        layerData.d_batchGradTensor = d_newBatchGradTensor;
                        zeroTensor(d_newBatchGradTensor, layerData.tensorFullSize);
                        return;
                    }
                }
                break;
            case ADAM_W:
                for (AdamWeightData& layerData : adamWParams) {
                    if (layerData.d_weightTensor == d_keyWeightTensor) {
                        layerData.d_batchGradTensor = d_newBatchGradTensor;
                        zeroTensor(d_newBatchGradTensor, layerData.tensorFullSize);
                        return;
                    }
                }
                break;
        }
    }


    // Updates all the registred weights using their registered alogrithm
    // Zeroes the gradient accumulation tensors after corresponding weight update
    // Can't be called when the layers are in inference mode
    void step(const uint batchSize) {
        for (AdamWeightData& layerData : adamParams) {
            adamOptimizerStep<<<ceilDiv(layerData.tensorFullSize, BLOCK_SIZE), BLOCK_SIZE>>>(
                layerData.d_weightTensor, layerData.d_oMMomentTensor, layerData.d_oVMomentTensor, 
                layerData.d_batchGradTensor, static_cast<float>(iteration),
                learningRate, beta1, beta2, epsilon,
                static_cast<float>(batchSize), layerData.tensorFullSize
            );
            checkCudaLastError();
            zeroTensor(layerData.d_batchGradTensor, layerData.tensorFullSize);
        }
        for (AdamWeightData& layerData : adamWParams) {
            adamWOptimizerStep<<<ceilDiv(layerData.tensorFullSize, BLOCK_SIZE), BLOCK_SIZE>>>(
                layerData.d_weightTensor, layerData.d_oMMomentTensor, layerData.d_oVMomentTensor,
                layerData.d_batchGradTensor, static_cast<float>(iteration),
                learningRate, beta1, beta2, weightDecay, epsilon,
                static_cast<float>(batchSize), layerData.tensorFullSize
            );
            checkCudaLastError();
            zeroTensor(layerData.d_batchGradTensor, layerData.tensorFullSize);
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
    }


    ~Optimizer() { reset(); }

};
