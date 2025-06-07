#pragma once
#include <curand_kernel.h>
#include "Utils.cuh"



/**
 * @brief Performs a tiled depthwise convolution forward pass.
 *
 * @tparam BLOCK_X_SIZE Number of threads in the x-dimension of a block.
 * @tparam BLOCK_Y_SIZE Number of threads in the y-dimension of a block.
 * @tparam FILTER_SIZE Size of the square convolution filter. Must be an odd number.
 * @tparam STRIDE Stride of the convolution operation.
 *
 * @param outTensor Output tensor in the format [B, C, H_out, W_out].
 * @param inTensor Input tensor in the format [B, C, H_in, W_in].
 * @param inFilters Convolution filters in the format [C, FILTER_SIZE, FILTER_SIZE].
 * @param CSize Number of channels.
 * @param outHSize Height of the output tensor.
 * @param outWSize Width of the output tensor.
 * @param outHBlocks Number of block rows in the output height dimension.
 * @param inHSize Height of the input tensor.
 * @param inWSize Width of the input tensor.
 *
 * @details
 * The grid for this kernel is launched with a 2D configuration:
 * - Grid x-dimension: ceil(outWSize / BLOCK_X_SIZE) blocks.
 * - Grid y-dimension: BSize * CSize * outHBlocks, where outHBlocks = ceil(outHSize / BLOCK_Y_SIZE).
 */
template <
    int BLOCK_X_SIZE,
    int BLOCK_Y_SIZE,
    int FILTER_SIZE,
    int STRIDE
> __global__
void depthwiseConvForward(
        float* __restrict__ outTensor,
        const float* __restrict__ inTensor,
        const float* __restrict__ inFilters,
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
        shFilter[loadIdx] = inFilters[CIdx * FILTER_FULL_SIZE + loadIdx];
    }
    __syncthreads();
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
        shTile[loadIdx] = inTensor[inLoadIdx];
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
    outTensor[tIdx] = convolutionSum;
}



/**
 * @brief Computes the gradient tensor dLoss/dI = dLoss/dO ∗ 180° rotated filter.
 *
 * @tparam BLOCK_X_SIZE Number of threads in the x-dimension of a block.
 * @tparam BLOCK_Y_SIZE Number of threads in the y-dimension of a block.
 * @tparam FILTER_SIZE Size of the original convolution filter.
 * @tparam STRIDE Stride of the original convolution operation.
 *
 * @param outTensor Output tensor dLoss/dI in the format [B, C, H_in, W_in].
 * @param inTensor Non-expanded input tensor dLoss/dO in the format [B, C, H_out, W_out].
 * @param inFilters Convolution filters in the format [C, FILTER_SIZE, FILTER_SIZE].
 * @param CSize Number of channels.
 * @param outHSize Height of the output tensor.
 * @param outWSize Width of the output tensor.
 * @param outHBlocks Number of block rows in the output height dimension.
 * @param inHSize Height of the input tensor.
 * @param inWSize Width of the input tensor.
 *
 * @details
 * The grid for this kernel is launched with a 2D configuration:
 * - Grid x-dimension: ceil(outWSize / BLOCK_X_SIZE) blocks.
 * - Grid y-dimension: BSize * CSize * outHBlocks, where outHBlocks = ceil(outHSize / BLOCK_Y_SIZE).
 */
template <
    int BLOCK_X_SIZE,
    int BLOCK_Y_SIZE,
    int FILTER_SIZE,
    int STRIDE
> __global__
void depthwiseConvBackward(
        float* __restrict__ outTensor,
        const float* __restrict__ inTensor,
        const float* __restrict__ inFilters,
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
    constexpr int EXPANDED_TILE_Y_SIZE = BLOCK_Y_SIZE + FILTER_SIZE - 1;
    constexpr int TILE_Y_SIZE = ceilDiv(EXPANDED_TILE_Y_SIZE, STRIDE);
    constexpr int EXPANDED_TILE_X_SIZE = BLOCK_X_SIZE + FILTER_SIZE - 1;
    constexpr int TILE_X_SIZE = ceilDiv(EXPANDED_TILE_X_SIZE, STRIDE);
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
        shFilter[loadIdx] = inFilters[CIdx * FILTER_FULL_SIZE + loadIdx];
    }
    __syncthreads();
    // Calculate in padding offset W and H first needed value indexes for the block
    const int inWFirstNeededIdx = ceilDiv(static_cast<int>(blockIdx.x) * BLOCK_X_SIZE - FILTER_PADD_SIZE, STRIDE);
    const int inHFirstNeededIdx = ceilDiv(static_cast<int>(blockIdx.y) % outHBlocks * BLOCK_Y_SIZE - FILTER_PADD_SIZE, STRIDE);
    // Calculate in B * C flat index part for input indexes
    const int BCIdx = BIdx * CSize + CIdx;
    const int inBCIdxPart = BCIdx * inHSize * inWSize;
    // Load the padded tiles into shared memory
    for (int loadIdx = tInBlockIdx; loadIdx < TILE_FULL_SIZE; loadIdx += THREADS_IN_BLOCK) {
        // Calculate in H and Y load indexes
        const int inLoadHIdx = inHFirstNeededIdx + loadIdx / TILE_X_SIZE;
        const int inLoadWIdx = inWFirstNeededIdx + loadIdx % TILE_X_SIZE;
        // Use zero padding if out of bounds
        if (inLoadHIdx < 0 || inLoadHIdx >= inHSize
                || inLoadWIdx < 0 || inLoadWIdx >= inWSize) {
            shTile[loadIdx] = 0.0f;
            continue;
        }
        // Calculate flat input index and load convoluted region cell to shared memory
        const int inLoadIdx = inBCIdxPart + inLoadHIdx * inWSize + inLoadWIdx;
        shTile[loadIdx] = inTensor[inLoadIdx];
    }
    __syncthreads();
    // Check bounds for the calculated indexes after loading data
    if (outHIdx >= outHSize || outWIdx >= outWSize) return;
    // Calculate top left convolution indexes for the expanded input
    const int expandedHIdx = outHIdx - FILTER_PADD_SIZE;
    const int expandedWIdx = outWIdx - FILTER_PADD_SIZE;
    // Compute inverted convolution on the filter and expanded input region
    float convolutionSum = 0.0f;
    for (int filterYIdx = 0; filterYIdx < FILTER_SIZE; ++filterYIdx) {
        for (int filterXIdx = 0; filterXIdx < FILTER_SIZE; ++filterXIdx) {
            // Calculate the current expanded input index offset by the filter
            const int expCurrHIdx = expandedHIdx + filterYIdx;
            const int expCurrWIdx = expandedWIdx + filterXIdx;
            // Check whether the expanded region is occupied with value or not
            if (expCurrHIdx % STRIDE || expCurrWIdx % STRIDE) continue;
            // Calculate where in the shared memory tile is the needed value
            const int tileYIdx = expCurrHIdx / STRIDE - inHFirstNeededIdx;
            const int tileXIdx = expCurrWIdx / STRIDE - inWFirstNeededIdx;
            const float regionValue = shTile[tileYIdx * TILE_X_SIZE + tileXIdx];
            // Load the corresponding inverted filter value
            const int invrtFilterYIdx = (FILTER_SIZE - filterYIdx - 1);
            const int invrtFilterXIdx = (FILTER_SIZE - filterXIdx - 1);
            const float filterValue = shFilter[invrtFilterYIdx * FILTER_SIZE + invrtFilterXIdx];
            // Multiply the filter value with the corresponding region cell
            convolutionSum += regionValue * filterValue;
        }
    }
    // Calculate flat output index and save result
    const int outBCIdxPart = BCIdx * outHSize * outWSize;
    const int tIdx = outBCIdxPart + outHIdx * outWSize + outWIdx;
    outTensor[tIdx] = convolutionSum;
}



/**
 * @brief Sums up the values for given warp 
 * @param threadValue Value of the single thread in the warp
 * @returns for first thread in the warp the total sum for the warp.
 */
__forceinline__ __device__ 
float warpReduce(float threadValue) {
    for (int offset = 16; offset > 0; offset /= 2) {
        threadValue += __shfl_down_sync(0xFFFFFFFF, threadValue, offset);
    }
    return threadValue; 
}



/**
 * @brief Computes the batch accumulated gradient tensor dLoss/dK.
 *
 * @tparam BLOCK_SIZE Number of threads in a single block.
 * @tparam FILTER_SIZE Size of the original convolution filter.
 * @tparam STRIDE Stride of the original convolution operation.
 *
 * @param outFilterGrads Output filter gradients in the format [C, FILTER_SIZE, FILTER_SIZE].
 * @param outTensor Output gradient tensor dLoss/dO in the format [B, C, H_out, W_out].
 * @param inTensor Input tensor dI in the format [B, C, H_in, W_in].
 * @param BSize Batch size.
 * @param CSize Number of channels.
 * @param outHWSize Product of height * width of the output tensor.
 * @param outWSize Width of the output tensor.
 * @param inHSize Height of the input tensor.
 * @param inWSize Width of the input tensor.
 *
 * @details
 * The grid for this kernel is launched with a 1D configuration:
 * - Grid x-dimension: CSize * H_fil * W_fil blocks.
 */
template <
    int BLOCK_SIZE,
    int FILTER_SIZE,
    int STRIDE
> __global__
void depthwiseConvBackwardGrad(
        float* __restrict__ outFilterGrads,
        const float* __restrict__ outTensor,
        const float* __restrict__ inTensor,
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
            const float outValue = outTensor[outIdx];
            const float inValue = inTensor[inIdx];
            threadValue += outValue * inValue;
        }
    }
    // Warp reduce within the whole block into shared memory
    threadValue = warpReduce(threadValue);
    if (tInWarpIdx == 0) {
        shResults[warpInBlockIdx] = threadValue;
    }
    __syncthreads();
    // Exit all warps except first
    if (warpInBlockIdx != 0) return;
    // Warp reduce the only first warp to get the final sum value
    threadValue = tInWarpIdx < WARPS_PER_BLOCK ? shResults[tInWarpIdx] : 0;
    threadValue = warpReduce(threadValue);
    // Save the final value for the whole block
    if (threadIdx.x != 0) return;
    outFilterGrads[blockIdx.x] = threadValue;
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
    else randVal = curand_uniform(&state)* 2.0f - 1.0f;
    outValues[tIdx] = randVal * range;
}



__global__
void adamOptimizerStep(
        float* __restrict__ weightTensor,
        float* __restrict__ mMomentTensor,
        float* __restrict__ vMomentTensor,
        const float* __restrict__ batchGradTensor,
        const float learningRate,
        const float iteration,
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
        const uint iteration,
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













//////////////////////// Maybe used maybe not ////////////////////////////
__global__
void transposeMatrix(
        float* __restrict__ matrixOut,
        const float* __restrict__ matrixIn,
        const uint colSize,
        const uint rowSize) {
    const uint tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= colSize * rowSize) return;
    const uint colIdx = tIdx / rowSize;
    const uint rowIdx = tIdx % rowSize;
    matrixOut[tIdx] = matrixIn[rowIdx * colSize + colIdx];
}


template <typename Operation>
__global__
void rowBroadcastOpInplace(
        float* __restrict__ outValues,
        const float* __restrict__ inVec,
        const uint rowSize,
        const uint size) {
    const uint tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= size) return;
    const uint rowIdx = tIdx % rowSize;
    const Operation operation;
    outValues[tIdx] = operation(outValues[tIdx], inVec[rowIdx]);
}


template <typename Operation>
__global__
void rowBroadcastOp(
        float* __restrict__ outValues,
        const float* __restrict__ inValues,
        const float* __restrict__ inVec,
        const uint rowSize,
        const uint size) {
    const uint tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= size) return;
    const uint rowIdx = tIdx % rowSize;
    const Operation operation;
    outValues[tIdx] = operation(inValues[tIdx], inVec[rowIdx]);
}


template <typename Operation>
__global__ 
void elementwiseOpInplace(
        float* __restrict__ tensor, 
        const uint size) {
    const uint tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= size) return;
    const Operation operation;
    tensor[tIdx] = operation(tensor[tIdx]);
}


template <typename Operation>
__global__
void elementwiseOp(
        float* __restrict__ outTensor,
        const float* __restrict__ inTensor,
        const uint size) {
    const uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) return;
    const Operation operation;
    outTensor[tid] = operation(inTensor[tid]);
}


template <typename Operation>
__global__ 
void elementwiseScalarOpInplace(
        float* __restrict__ tensor,
        const float value,
        const uint size) {
    const uint tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= size) return;
    const Operation operation;
    tensor[tIdx] = operation(tensor[tIdx], value);
}


template <typename Operation>
__global__
void elementwiseScalarOp(
        float* __restrict__ outTensor,
        const float* __restrict__ inTensor,
        const float value,
        const uint size) {
    const uint tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= size) return;
    const Operation operation;
    outTensor[tIdx] = operation(inTensor[tIdx], value);
}


template <typename Operation>
__global__ 
void elementwise2TensorOpInplace(
        float* __restrict__ toTensor, 
        const float* __restrict__ fromTensor, 
        const uint size) {
    const uint tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= size) return;
    const Operation operation;
    toTensor[tIdx] = operation(toTensor[tIdx], fromTensor[tIdx]);
}


template <typename Operation>
__global__ 
void elementwise2TensorOp(
        float* __restrict__ outTensor,
        const float* __restrict__ tensorA,
        const float* __restrict__ tensorB, 
        const uint size) {
    const uint tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= size) return;
    const Operation operation;
    outTensor[tIdx] = operation(tensorA[tIdx], tensorB[tIdx]);
}


template <uint TensorDim>
__global__
void reshapeTensor(
        float* __restrict__ tensorOut,
        const float* __restrict__ tensorIn,
        const uint outRowSize,
        const uint outOrder[TensorDim],
        const uint outDimSizeSums[TensorDim],
        const uint inDimSizeSums[TensorDim]) {
    const uint outFullIdx = blockIdx.x * blockDim.x + threadIdx.x;
    uint outIdxes[TensorDim];
    uint outCurrIdx = outFullIdx;
    for (uint dimIdx = 0; dimIdx < TensorDim; ++dimIdx) {
        const uint currentDimSizeSum = outDimSizeSums[dimIdx];
        outIdxes[dimIdx] = outCurrIdx / currentDimSizeSum;
        outCurrIdx = outCurrIdx % currentDimSizeSum;
    }
    float value = 0.0f;
    if (outIdxes[TensorDim - 1] < outRowSize) {
        uint inFullIdx = 0;
        for (uint dimIdx = 0; dimIdx < TensorDim; ++dimIdx) {
            const uint currDimIdx = outIdxes[outOrder[dimIdx]];
            inFullIdx += currDimIdx * inDimSizeSums[currDimIdx];
        }
        value = tensorIn[inFullIdx];
    }
    tensorOut[outFullIdx] = value;
}
