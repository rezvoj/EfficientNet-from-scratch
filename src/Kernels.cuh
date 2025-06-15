#pragma once
#include <curand_kernel.h>
#include "Utils.cuh"



// Performs a tiled depthwise convolution forward pass
// The grid for this kernel is launched with a 2D configuration:
//  x-dimension: ceil(outWSize / BLOCK_X_SIZE) blocks
//  y-dimension: BSize * CSize * outHBlocks, where outHBlocks = ceil(outHSize / BLOCK_Y_SIZE)
template <
    int BLOCK_X_SIZE,
    int BLOCK_Y_SIZE,
    int FILTER_SIZE,
    int STRIDE
> __global__
void depthwiseConvForward(
        float* __restrict__ outputTensor,
        const float* __restrict__ inputTensor,
        const float* __restrict__ filtersTensor,
        const int CSize,
        const int outHSize,
        const int outWSize,
        const int outHBlocks,
        const int inHSize,
        const int inWSize) {
    static_assert(FILTER_SIZE % 2);
    // Calculate template inferred constants
    constexpr int THREADS_IN_BLOCK = BLOCK_Y_SIZE * BLOCK_X_SIZE;
    // Calculate filter size and filter padding
    constexpr int FILTER_FULL_SIZE = FILTER_SIZE * FILTER_SIZE;
    constexpr int FILTER_PADD_SIZE = FILTER_SIZE / 2;
    // Calculate the maximal tile sizes needed
    constexpr int TILE_SIZE_FIX = 2 * FILTER_PADD_SIZE - (STRIDE - 1);
    constexpr int TILE_Y_SIZE = BLOCK_Y_SIZE * STRIDE + TILE_SIZE_FIX;
    constexpr int TILE_X_SIZE = BLOCK_X_SIZE * STRIDE + TILE_SIZE_FIX;
    constexpr int TILE_FULL_SIZE = TILE_Y_SIZE * TILE_X_SIZE;
    // Define shared memory for filter an tile
    __shared__ float shFilter[FILTER_FULL_SIZE];
    __shared__ float shTile[TILE_FULL_SIZE];
    // Calculate W and flat B * C * H out index
    const int outWIdx = blockIdx.x * BLOCK_X_SIZE + threadIdx.x;
    const int outBCHIdx = blockIdx.y * BLOCK_Y_SIZE + threadIdx.y;
    // Calculate flat within block index
    const int tInBlockIdx = threadIdx.y * BLOCK_X_SIZE + threadIdx.x;
    // Calculate helper divisor constants
    const int outHBlockSize = outHBlocks * BLOCK_Y_SIZE;
    const int outCHBlockSize = CSize * outHBlockSize;
    // Deconstruct the flat index into B, C and H out indexes
    const int BIdx = outBCHIdx / outCHBlockSize;
    const int outLeftoverIdx = outBCHIdx % outCHBlockSize;
    const int CIdx = outLeftoverIdx / outHBlockSize;
    const int outHIdx = outLeftoverIdx % outHBlockSize;
    // Load convolution filter to the shared memory
    for (int loadIdx = tInBlockIdx; loadIdx < FILTER_FULL_SIZE; loadIdx += THREADS_IN_BLOCK) {
        shFilter[loadIdx] = filtersTensor[CIdx * FILTER_FULL_SIZE + loadIdx];
    }
    // Calculate in padding offset W and H start indexes for the block
    const int inWStartOffsetIdx = blockIdx.x * BLOCK_X_SIZE * STRIDE - FILTER_PADD_SIZE;
    const int inHStartOffsetIdx = blockIdx.y % outHBlocks * BLOCK_Y_SIZE * STRIDE - FILTER_PADD_SIZE;
    // Calculate in B * C flat index part for input indexes
    const int BCIdx = BIdx * CSize + CIdx;
    const int inBCIdxPart = BCIdx * inHSize * inWSize;
    // Load the padded tiles into shared memory
    for (int loadIdx = tInBlockIdx; loadIdx < TILE_FULL_SIZE; loadIdx += THREADS_IN_BLOCK) {
        // Calculate in H and Y load indexes
        const int inLoadHIdx = inHStartOffsetIdx + loadIdx / TILE_X_SIZE;
        const int inLoadWIdx = inWStartOffsetIdx + loadIdx % TILE_X_SIZE;
        // Use zero padding if out of bounds
        if (inLoadHIdx < 0 || inLoadHIdx >= inHSize
                || inLoadWIdx < 0 || inLoadWIdx >= inWSize) {
            shTile[loadIdx] = 0.0f;
            continue;
        }
        // Calculate flat input index and load convoluted region cell to shared memory
        const int inLoadIdx = inBCIdxPart + inLoadHIdx * inWSize + inLoadWIdx;
        shTile[loadIdx] = inputTensor[inLoadIdx];
    }
    __syncthreads();
    // Check bounds for the calculated indexes after loading data
    if (outHIdx >= outHSize || outWIdx >= outWSize) return;
    // Calculate top left Y and X convolution indexes within the block
    const int tileYIdx = threadIdx.y * STRIDE;
    const int tileXIdx = threadIdx.x * STRIDE;
    // Compute convolution on the filter and region
    float convolutionSum = 0.0f;
    for (int filterYIdx = 0; filterYIdx < FILTER_SIZE; ++filterYIdx) {
        for (int filterXIdx = 0; filterXIdx < FILTER_SIZE; ++filterXIdx) {
            // Load the corresponding filter value
            const float filterValue = shFilter[filterYIdx * FILTER_SIZE + filterXIdx];
            // Calculate index of the corresponding cell
            const int convIdxYPart = (tileYIdx + filterYIdx) * TILE_X_SIZE;
            const int convIdxXPart = tileXIdx + filterXIdx;
            // Multiply the filter value with the corresponding region cell
            convolutionSum += shTile[convIdxYPart + convIdxXPart] * filterValue;
        }
    }
    // Calculate flat output index and save result
    const int outBCIdxPart = BCIdx * outHSize * outWSize;
    const int tIdx = outBCIdxPart + outHIdx * outWSize + outWIdx;
    outputTensor[tIdx] = convolutionSum;
}



// Computes the gradient tensor dLoss/dI from dLoss/dO and filters
// The grid for this kernel is launched with a 2D configuration:
//  grid x-dimension: ceil(inWSize / BLOCK_X_SIZE) blocks
//  grid y-dimension: BSize * CSize * inHBlocks, where inHBlocks = ceil(inHSize / BLOCK_Y_SIZE)
template <
    int BLOCK_X_SIZE,
    int BLOCK_Y_SIZE,
    int FILTER_SIZE,
    int STRIDE
> __global__
void depthwiseConvBackward(
        float* __restrict__ inputGradTensor,
        const float* __restrict__ outputGradTensor,
        const float* __restrict__ filtersTensor,
        const int CSize,
        const int inHSize,
        const int inWSize,
        const int inHBlocks,
        const int outHSize,
        const int outWSize) {
    static_assert(FILTER_SIZE % 2);
    // Calculate template inferred constants
    constexpr int THREADS_IN_BLOCK = BLOCK_Y_SIZE * BLOCK_X_SIZE;
    // Calculate filter size and filter padding
    constexpr int FILTER_FULL_SIZE = FILTER_SIZE * FILTER_SIZE;
    constexpr int FILTER_PADD_SIZE = FILTER_SIZE / 2;
    // Calculate the maximal tile sizes needed
    constexpr int EXPANDED_TILE_Y_SIZE = BLOCK_Y_SIZE + FILTER_SIZE - 1;
    constexpr int TILE_Y_SIZE = ceilDiv(EXPANDED_TILE_Y_SIZE, STRIDE);
    constexpr int EXPANDED_TILE_X_SIZE = BLOCK_X_SIZE + FILTER_SIZE - 1;
    constexpr int TILE_X_SIZE = ceilDiv(EXPANDED_TILE_X_SIZE, STRIDE);
    constexpr int TILE_FULL_SIZE = TILE_Y_SIZE * TILE_X_SIZE;
    // Define shared memory for filter an tile
    __shared__ float shFilter[FILTER_FULL_SIZE];
    __shared__ float shTile[TILE_FULL_SIZE];
    // Calculate W and flat B * C * H in index
    const int inWIdx = blockIdx.x * BLOCK_X_SIZE + threadIdx.x;
    const int inBCHIdx = blockIdx.y * BLOCK_Y_SIZE + threadIdx.y;
    // Calculate flat within block index
    const int tInBlockIdx = threadIdx.y * BLOCK_X_SIZE + threadIdx.x;
    // Calculate helper divisor constants
    const int inHBlockSize = inHBlocks * BLOCK_Y_SIZE;
    const int inCHBlockSize = CSize * inHBlockSize;
    // Deconstruct the flat index into B, C and H in indexes
    const int BIdx = inBCHIdx / inCHBlockSize;
    const int inLeftoverIdx = inBCHIdx % inCHBlockSize;
    const int CIdx = inLeftoverIdx / inHBlockSize;
    const int inHIdx = inLeftoverIdx % inHBlockSize;
    // Load convolution filter to the shared memory
    for (int loadIdx = tInBlockIdx; loadIdx < FILTER_FULL_SIZE; loadIdx += THREADS_IN_BLOCK) {
        shFilter[loadIdx] = filtersTensor[CIdx * FILTER_FULL_SIZE + loadIdx];
    }
    // Calculate out padding offset W and H first needed value indexes for the block
    const int outWFirstNeededIdx = ceilDiv(static_cast<int>(blockIdx.x) * BLOCK_X_SIZE - FILTER_PADD_SIZE, STRIDE);
    const int outHFirstNeededIdx = ceilDiv(static_cast<int>(blockIdx.y) % inHBlocks * BLOCK_Y_SIZE - FILTER_PADD_SIZE, STRIDE);
    // Calculate out B * C flat index part for output indexes
    const int BCIdx = BIdx * CSize + CIdx;
    const int outBCIdxPart = BCIdx * outHSize * outWSize;
    // Load the padded tiles into shared memory
    for (int loadIdx = tInBlockIdx; loadIdx < TILE_FULL_SIZE; loadIdx += THREADS_IN_BLOCK) {
        // Calculate out H and Y load indexes
        const int outLoadHIdx = outHFirstNeededIdx + loadIdx / TILE_X_SIZE;
        const int outLoadWIdx = outWFirstNeededIdx + loadIdx % TILE_X_SIZE;
        // Use zero padding if out of bounds
        if (outLoadHIdx < 0 || outLoadHIdx >= outHSize
                || outLoadWIdx < 0 || outLoadWIdx >= outWSize) {
            shTile[loadIdx] = 0.0f;
            continue;
        }
        // Calculate flat output index and load convoluted region cell to shared memory
        const int outLoadIdx = outBCIdxPart + outLoadHIdx * outWSize + outLoadWIdx;
        shTile[loadIdx] = outputGradTensor[outLoadIdx];
    }
    __syncthreads();
    // Check bounds for the calculated indexes after loading data
    if (inHIdx >= inHSize || inWIdx >= inWSize) return;
    // Calculate top left convolution indexes for the expanded output
    const int expandedHIdx = inHIdx - FILTER_PADD_SIZE;
    const int expandedWIdx = inWIdx - FILTER_PADD_SIZE;
    // Compute inverted convolution on the filter and expanded output region
    float convolutionSum = 0.0f;
    for (int filterYIdx = 0; filterYIdx < FILTER_SIZE; ++filterYIdx) {
        for (int filterXIdx = 0; filterXIdx < FILTER_SIZE; ++filterXIdx) {
            // Calculate the current expanded output index offset by the filter
            const int expCurrHIdx = expandedHIdx + filterYIdx;
            const int expCurrWIdx = expandedWIdx + filterXIdx;
            // Check whether the expanded region is occupied with value or not
            if (expCurrHIdx % STRIDE || expCurrWIdx % STRIDE) continue;
            // Calculate where in the shared memory tile is the needed value
            const int tileYIdx = expCurrHIdx / STRIDE - outHFirstNeededIdx;
            const int tileXIdx = expCurrWIdx / STRIDE - outWFirstNeededIdx;
            const float regionValue = shTile[tileYIdx * TILE_X_SIZE + tileXIdx];
            // Load the corresponding inverted filter value
            const int invrtFilterYIdx = (FILTER_SIZE - filterYIdx - 1);
            const int invrtFilterXIdx = (FILTER_SIZE - filterXIdx - 1);
            const float filterValue = shFilter[invrtFilterYIdx * FILTER_SIZE + invrtFilterXIdx];
            // Multiply the filter value with the corresponding region cell
            convolutionSum += regionValue * filterValue;
        }
    }
    // Calculate flat input index and save result
    const int inBCIdxPart = BCIdx * inHSize * inWSize;
    const int tIdx = inBCIdxPart + inHIdx * inWSize + inWIdx;
    inputGradTensor[tIdx] = convolutionSum;
}



__forceinline__ __device__ 
float warpReduceSum(float threadValue) {
    for (int offset = 16; offset > 0; offset /= 2) {
        threadValue += __shfl_down_sync(0xFFFFFFFF, threadValue, offset);
    }
    return threadValue; 
}



// Accumulates the batch accumulated gradient tensor dLoss/dK
// The grid for this kernel is launched with a 1D configuration:
//  x-dimension: CSize * H_fil * W_fil blocks
template <
    int BLOCK_SIZE,
    int FILTER_SIZE,
    int STRIDE
> __global__
void depthwiseConvBackwardGrad(
        float* __restrict__ filterGrads,
        const float* __restrict__ outputGradTensor,
        const float* __restrict__ inputTensor,
        const int BSize,
        const int CSize,
        const int outHWSize,
        const int outWSize,
        const int inHSize,
        const int inWSize) {
    static_assert(FILTER_SIZE % 2);
    constexpr int WARP_SIZE = 32;
    static_assert(BLOCK_SIZE % WARP_SIZE == 0);
    // Calculate template inferred constants
    constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;
    // Calculate filter size and filter padding
    constexpr int FILTER_FULL_SIZE = FILTER_SIZE * FILTER_SIZE;
    constexpr int FILTER_PADD_SIZE = FILTER_SIZE / 2;
    // Define for sharing results between warps
    __shared__ float shResults[WARPS_PER_BLOCK];
    // Deconstruct the thread within block index to warp and thread withing warp indexes
    const int warpInBlockIdx = static_cast<int>(threadIdx.x) / WARP_SIZE;
    const int tInWarpIdx = static_cast<int>(threadIdx.x) % WARP_SIZE;
    // Deconstruct the block index to C, H_fil, W_fil indexes
    const int CIdx = static_cast<int>(blockIdx.x) / FILTER_FULL_SIZE;
    const int filterHWIdx = static_cast<int>(blockIdx.x) % FILTER_FULL_SIZE;
    const int filterOffsetHIdx = filterHWIdx / FILTER_SIZE - FILTER_PADD_SIZE;
    const int filterOffsetWIdx = filterHWIdx % FILTER_SIZE - FILTER_PADD_SIZE;
    // Sum the contribution values for the given thread
    float threadValue = 0.0f;
    for (int BIdx = warpInBlockIdx; BIdx < BSize; BIdx += WARPS_PER_BLOCK) {
        // Calculate B * C index part for input and output tensors
        const int BCIdx = BIdx * CSize + CIdx;
        const int outBCIdxPart = BCIdx * outHWSize;
        const int inBCIdxPart = BCIdx * inHSize * inWSize;
        // Iterate the contributions in the current batch
        for (int outHWIdx = tInWarpIdx; outHWIdx < outHWSize; outHWIdx += WARP_SIZE) {
            // Calculate contribution input and output tensor indexes
            const int outHIdx = outHWIdx / outWSize;
            const int outWIdx = outHWIdx % outWSize;
            const int inHIdx = outHIdx * STRIDE + filterOffsetHIdx;
            const int inWIdx = outWIdx * STRIDE + filterOffsetWIdx;
            // Check if padding or not was used during forward conv for this contribution
            if (inHIdx < 0 || inHIdx >= inHSize || inWIdx < 0 || inWIdx >= inWSize) continue;
            // Calculate contribution global input and output tensor indexes
            const int outIdx = outBCIdxPart + outHIdx * outWSize + outWIdx;
            const int inIdx = inBCIdxPart + inHIdx * inWSize + inWIdx;
            // Add the contribution value to the thread's sum
            const float outValue = outputGradTensor[outIdx];
            const float inValue = inputTensor[inIdx];
            threadValue += outValue * inValue;
        }
    }
    // Warp reduce within the whole block into shared memory
    threadValue = warpReduceSum(threadValue);
    if (tInWarpIdx == 0) {
        shResults[warpInBlockIdx] = threadValue;
    }
    __syncthreads();
    // Exit all warps except first
    if (warpInBlockIdx != 0) return;
    // Warp reduce the only first warp to get the final sum value
    threadValue = tInWarpIdx < WARPS_PER_BLOCK ? shResults[tInWarpIdx] : 0;
    threadValue = warpReduceSum(threadValue);
    // Accumulate the final value for the whole block
    if (threadIdx.x != 0) return;
    filterGrads[blockIdx.x] += threadValue;
}



__global__
void initValues(
        float* __restrict__ outValues,
        const float value,
        const uint size) {
    const uint tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= size) return;
    outValues[tIdx] = value;
}



template <bool UseNormal>
__global__
void initRandomValues(
        float* __restrict__ outValues,
        const uint seed,
        const uint offset,
        const float range,
        const uint size) {
    const uint tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= size) return;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, tIdx, offset, &state);
    float randVal;
    if constexpr (UseNormal) randVal = curand_normal(&state);
    else randVal = curand_uniform(&state) * 2.0f - 1.0f;
    outValues[tIdx] = randVal * range;
}



__global__
void adamOptimizerStep(
        float* __restrict__ weightTensor,
        float* __restrict__ mMomentTensor,
        float* __restrict__ vMomentTensor,
        const float* __restrict__ batchGradTensor,
        const float iteration,
        const float learningRate,
        const float beta1,
        const float beta2,
        const float epsilon,
        const float batchSize,
        const uint size) {
    const uint tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= size) return;
    // Calculate mean gradient for the batch
    const float gradValue = batchGradTensor[tIdx] / batchSize;
    // Calculate m_t = β1 * m_t-1 + (1 - β1) * g_t
    const float mMomentUpdated = beta1 * mMomentTensor[tIdx] + (1 - beta1) * gradValue;
    mMomentTensor[tIdx] = mMomentUpdated;
    // Calculate v_t = β2 * v_t-1 + (1 - β2) * g_t^2
    const float gradValueSquared = gradValue * gradValue;
    const float vMomentUpdated = beta2 * vMomentTensor[tIdx] + (1 - beta2) * gradValueSquared;
    vMomentTensor[tIdx] = vMomentUpdated;
    // Calculate m^_t = m_t / (1 - β1^t)
    const float mMomentHat = mMomentUpdated / (1 - powf(beta1, iteration));
    // Calculate v^_t = v_t / (1 - β2^t)
    const float vMomentHat = vMomentUpdated / (1 - powf(beta2, iteration));
    // Calculate θ_t = θ_t-1 - η * m^_t / (sqrt(v^_t) + ε)
    weightTensor[tIdx] -= learningRate * mMomentHat / (sqrtf(vMomentHat) + epsilon);
}



__global__
void adamWOptimizerStep(
        float* __restrict__ weightTensor,
        float* __restrict__ mMomentTensor,
        float* __restrict__ vMomentTensor,
        const float* __restrict__ batchGradTensor,
        const float iteration,
        const float learningRate,
        const float beta1,
        const float beta2,
        const float weightDecay,
        const float epsilon,
        const uint batchSize,
        const uint size) {
    const uint tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= size) return;
    // Calculate mean gradient for the batch
    const float gradValue = batchGradTensor[tIdx] / batchSize;
    // Calculate m_t = β1 * m_t-1 + (1 - β1) * g_t
    const float mMomentUpdated = beta1 * mMomentTensor[tIdx] + (1 - beta1) * gradValue;
    mMomentTensor[tIdx] = mMomentUpdated;
    // Calculate v_t = β2 * v_t-1 + (1 - β2) * g_t^2
    const float gradValueSquared = gradValue * gradValue;
    const float vMomentUpdated = beta2 * vMomentTensor[tIdx] + (1 - beta2) * gradValueSquared;
    vMomentTensor[tIdx] = vMomentUpdated;
    // Calculate m^_t = m_t / (1 - β1^t)
    const float mMomentHat = mMomentUpdated / (1 - powf(beta1, iteration));
    // Calculate v^_t = v_t / (1 - β2^t)
    const float vMomentHat = vMomentUpdated / (1 - powf(beta2, iteration));
    // Calculate θ_t = θ_t-1 - η * (m^_t / (sqrt(v^_t) + ε) + λ * θ_t-1)
    const float prevWeight = weightTensor[tIdx];
    const float gradPart = mMomentHat / (sqrtf(vMomentHat) + epsilon);
    const float decayPart = weightDecay * prevWeight;
    weightTensor[tIdx] -= learningRate * (gradPart + decayPart);
}



__global__
void elementwiseAddInplace(
        float* __restrict__ outTensor,
        const float* __restrict__ tensor,
        const uint size) {
    const uint tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= size) return;
    outTensor[tIdx] += tensor[tIdx];
};



__global__
void elementwiseMul(
        float* __restrict__ outTensor,
        const float* __restrict__ tensorA,
        const float* __restrict__ tensorB,
        const uint size) {
    const uint tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= size) return;
    outTensor[tIdx] = tensorA[tIdx] * tensorB[tIdx];
};



__global__
void elementwiseMulInplace(
        float* __restrict__ tensorA,
        const float* __restrict__ inTensorB,
        const uint size) {
    const uint tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= size) return;
    tensorA[tIdx] *= inTensorB[tIdx];
};



__global__
void elementwiseMulBackwardInplace(
        float* __restrict__ tensorA,
        float* __restrict__ tensorB,
        const float* __restrict__ inGradTensor,
        const uint size) {
    const uint tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= size) return;
    const float valueA = tensorA[tIdx];
    const float valueB = tensorB[tIdx];
    const float ValueGrad = inGradTensor[tIdx];
    tensorA[tIdx] = ValueGrad * valueB;
    tensorB[tIdx] = ValueGrad * valueA;
};



__global__ 
void elementwiseScalarMulInplace(
        float* __restrict__ tensor,
        const float value,
        const uint size) {
    const uint tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= size) return;
    tensor[tIdx] *= value;
}



struct ReLU { 
    __forceinline__ __device__
    static float forward(const float x) {
        return fmaxf(0.0f, x);
    }
    __forceinline__ __device__
    static float backward(const float x, [[maybe_unused]] const float y) {
        return (x > 0.0f) ? 1.0f : 0.0f;
    }
};



struct Sigmoid {
    __forceinline__ __device__
    static float forward(const float x) {
        return 1.0f / (1.0f + expf(-x));
    }
    __forceinline__ __device__
    static float backward([[maybe_unused]] const float x, const float y) {
        return y * (1.0f - y);
    }
};



struct SiLU {
    __forceinline__ __device__
    static float forward(const float x) {
        return x / (1.0f + expf(-x));
    }
    __forceinline__ __device__
    static float backward(const float x, const float y) {
        const float sig = (x == 0.0f) ? 0.5f : y / x;
        return sig + x * sig * (1.0f - sig);
    }
};



template <typename ACTIVATION>
__global__ 
void elementwiseActivation(
        float* __restrict__ outTensor,
        const float* __restrict__ inTensor,
        const uint size) {
    const uint tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= size) return;
    outTensor[tIdx] = ACTIVATION::forward(inTensor[tIdx]);
}



template <typename ACTIVATION>
__global__ 
void elementwiseActivationInplace(
        float* __restrict__ tensor,
        const uint size) {
    const uint tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= size) return;
    tensor[tIdx] = ACTIVATION::forward(tensor[tIdx]);
}



template <typename ACTIVATION>
__global__ 
void elementwiseActivationBackwardInplace(
        float* __restrict__ outTensor,
        const float* __restrict__ inputTensor,
        const uint size) {
    const uint tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= size) return;
    const float x = inputTensor[tIdx];
    const float y = ACTIVATION::forward(x);
    outTensor[tIdx] *= ACTIVATION::backward(x, y);
}



__global__ 
void tensorExpansion(
        float* __restrict__ outTensor,
        const float* __restrict__ inTensor,
        const float scale,
        const uint fullSize,
        const uint outHWSize) {
    const uint tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= fullSize) return;
    const uint inIdx = tIdx / outHWSize;
    outTensor[tIdx] = scale * inTensor[inIdx];
}



__global__
void crossEntropyLossGradInplace(
        float* __restrict__ probsToLossGradTensor,
        const uint* __restrict__ inLabels,
        const uint batchSize,
        const uint numCategories) {
    const uint tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= batchSize * numCategories) return;
    const uint batchIdx = tIdx / numCategories;
    const uint catIdx = tIdx % numCategories;
    const uint label = inLabels[batchIdx];
    const float trueValue = static_cast<float>(label == catIdx);
    const float probValue = probsToLossGradTensor[tIdx];
    probsToLossGradTensor[tIdx] = probValue - trueValue;
}



__global__
void crossEntropyLoss(
        float* __restrict__ outLossTensor,
        const float* __restrict__ inProbsTensor,
        const uint* __restrict__ inLabels,
        const float epsilon,
        const uint batchSize,
        const uint numCategories) {
    const uint batchIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batchIdx >= batchSize) return;
    const uint probIdx = batchIdx * numCategories + inLabels[batchIdx];
    const float probValue = inProbsTensor[probIdx];
    outLossTensor[batchIdx] = -logf(probValue + epsilon);
}



__global__
void broadcastAddBiasInplace(
        float* __restrict__ outMatrix,
        const float* __restrict__ inBiases,
        const uint CSize,
        const uint size) {
    const uint tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= size) return;
    const uint biasIdx = tIdx % CSize;
    outMatrix[tIdx] += inBiases[biasIdx];
}
