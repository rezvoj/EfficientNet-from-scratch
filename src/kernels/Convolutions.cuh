#pragma once
#include "../utils/Helpers.cuh"



/**
 * @brief Creates 2nd matrix intermediate im2col representation for convolution forward pass.
 *
 * @tparam FILTER_SIZE Size of the convolution filter. Must be an odd number.
 * @tparam STRIDE Stride of the convolution operation.
 *
 * @param tensorOut Output tensor in the format [C_in * FILTER_SIZE * FILTER_SIZE, B * H_out * W_out].
 * @param tensorIn Input tensor in the format [C_in, B, H_in, W_in].
 * @param CSize Number of input channels (C_in).
 * @param BSize Number of batches.
 * @param outHWSize Product of the output height and width (H_out * W_out).
 * @param outWSize Width of the output tensor (W_out).
 * @param inHSize Height of the input tensor (H_in).
 * @param inWSize Width of the input tensor (W_in).
 *
 * @details
 * Performs im2col transformation preparing the second matrix for the following GEMM:
 * (C_out, B * H_out * W_out) = (C_out, C_in * H_fil * W_fil) @ (C_in * H_fil * W_fil, B * H_out * W_out)
 */
template <int FILTER_SIZE, int STRIDE>
static __global__ void im2colConv(
        float* __restrict__ tensorOut,
        const float* __restrict__ tensorIn,
        const int CSize,
        const int BSize,
        const int outHWSize,
        const int outWSize,
        const int inHSize,
        const int inWSize) {
    static_assert(FILTER_SIZE % 2);
    // Calculate filter size and filter padding
    constexpr int FILTER_FULL_SIZE = FILTER_SIZE * FILTER_SIZE;
    constexpr int FILTER_PADD_SIZE = FILTER_SIZE / 2;
    // Calculate base indexes and check bounds
    const int outBHWIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const int filterCHWIdx = blockIdx.y * blockDim.y + threadIdx.y;
    const int outBHWSize = BSize * outHWSize;
    if (outBHWIdx >= outBHWSize || filterCHWIdx >= CSize * FILTER_FULL_SIZE) return;
    const int tIdx = filterCHWIdx * outBHWSize + outBHWIdx;
    // Calculate the C index and flat filter H * W index
    const int CIdx = filterCHWIdx / FILTER_FULL_SIZE;
    const int filterHWIdx = filterCHWIdx % FILTER_FULL_SIZE;
    // Calculate filter offset from convolution anchor point
    const int filterHOffsetIdx = filterHWIdx / FILTER_SIZE - FILTER_PADD_SIZE;
    const int filterWOffsetIdx = filterHWIdx % FILTER_SIZE - FILTER_PADD_SIZE;
    // Calculate the B index and flat out H * W index
    const int BIdx = outBHWIdx / outHWSize;
    const int outHWIdx = outBHWIdx % outHWSize;
    // Calculate the patch point aka the offset anchor point
    const int inPointHIdx = outHWIdx / outWSize * STRIDE + filterHOffsetIdx;
    const int inPointWIdx = outHWIdx % outWSize * STRIDE + filterWOffsetIdx;
    // Load the patch point's value if it isn't OOB
    float pointValue = 0.0f;
    if (inPointHIdx >= 0 && inPointHIdx < inHSize 
            && inPointWIdx >= 0 && inPointWIdx < inWSize) {
        const int CBIdx = CIdx * BSize + BIdx;
        const int inCBIdxPart = CBIdx * inHSize * inWSize;
        pointValue = tensorIn[inCBIdxPart + inPointHIdx * inWSize + inPointWIdx];
    }
    tensorOut[tIdx] = pointValue;
}



/**
 * @brief Creates 2nd matrix intermediate im2col representation for computing dLoss/dK.
 *
 * @tparam FILTER_SIZE Size of the original convolution filter.
 * @tparam STRIDE Stride of the original convolution operation.
 *
 * @param tensorOut Output tensor in the format [B * H_out * W_out, C_in * FILTER_SIZE * FILTER_SIZE].
 * @param tensorIn Input tensor in the format [C_in, B, H_in, W_in].
 * @param CSize Number of channels.
 * @param BSize Number of batches.
 * @param outHWSize Product of the output height and width (H_out * W_out).
 * @param outWSize Width of the output tensor (W_out).
 * @param inHSize Height of the input tensor (H_in).
 * @param inWSize Width of the input tensor (W_in).
 *
 * @details
 * Performs im2col transformation preparing the second matrix for the following GEMM:
 * - (C_out, C_in * H_fil * W_fil) = (C_out, B * H_out * W_out) @ (B * H_out * W_out, C_in * H_fil * W_fil)
 *
 * The grid for this kernel is launched with a 2D configuration:
 * - Grid x-dimension: covering C_in * FILTER_SIZE * FILTER_SIZE elements.
 * - Grid y-dimension: covering B * H_out * W_out elements.
 */
template <int FILTER_SIZE, int STRIDE>
static __global__ void im2colConvBackward(
        float* __restrict__ tensorOut,
        const float* __restrict__ tensorIn,
        const int CSize,
        const int BSize,
        const int outHWSize,
        const int outWSize,
        const int inHSize,
        const int inWSize) {
    static_assert(FILTER_SIZE % 2);
    // Calculate filter size and filter padding
    constexpr int FILTER_FULL_SIZE = FILTER_SIZE * FILTER_SIZE;
    constexpr int FILTER_PADD_SIZE = FILTER_SIZE / 2;
    // Calculate base indexes and check bounds
    const int filterCHWIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const int outBHWIdx = blockIdx.y * blockDim.y + threadIdx.y;
    const int filterCHWSize = CSize * FILTER_FULL_SIZE;
    const int outBHWSize = BSize * outHWSize;
    if (filterCHWIdx >= filterCHWSize || outBHWIdx >= outBHWSize) return;
    const int tIdx = outBHWIdx * filterCHWSize + filterCHWIdx;
    // Calculate B and flat out H * W indexes
    const int BIdx = outBHWIdx / outHWSize;
    const int outHWIdx = outBHWIdx % outHWSize;
    // Calculate the C index and flat filter H * W indexes
    const int CIdx = filterCHWIdx / FILTER_FULL_SIZE;
    const int filterHWIdx = filterCHWIdx % FILTER_FULL_SIZE;
    // Calculate filter offset from anchor point
    const int filterHOffsetIdx = filterHWIdx / FILTER_SIZE - FILTER_PADD_SIZE;
    const int filterWOffsetIdx = filterHWIdx % FILTER_SIZE - FILTER_PADD_SIZE;
    // Calculate the virtual patch point
    const int inPointHIdx = outHWIdx / outWSize * STRIDE + filterHOffsetIdx;
    const int inPointWIdx = outHWIdx % outWSize * STRIDE + filterWOffsetIdx;
    // Load the patch point's value if it isn't OOB
    float pointValue = 0.0f;
    if (inPointHIdx >= 0 && inPointHIdx < inHSize 
            && inPointWIdx >= 0 && inPointWIdx < inWSize) {
        // Construct flat in C * B * H * W index
        const int CBIdx = CIdx * BSize + BIdx;
        const int inCBIdxPart = CBIdx * inHSize * inWSize;
        const int inHWIndex = inPointHIdx * inWSize + inPointWIdx;
        pointValue = tensorIn[inCBIdxPart + inHWIndex];
    }
    tensorOut[tIdx] = pointValue;
}



/**
 * @brief Creates 1st matrix intermediate im2col representation for computing dLoss/dI.
 *
 * @tparam FILTER_SIZE Size of the convolution filter.
 *
 * @param tensorOut Output tensor in the format [C_in, C_out * FILTER_SIZE * FILTER_SIZE] with flipped filter weights.
 * @param tensorIn Input tensor in the format [C_out, C_in * FILTER_SIZE * FILTER_SIZE].
 * @param inCSize Number of input channels (C_in).
 * @param outCSize Number of output channels (C_out).
 *
 * @details
 * Performs im2col transformation preparing the first matrix for the following GEMM:
 * - (C_in, B * H_in * W_in) = (C_in, C_out * H_fil * W_fil) @ (C_out * H_fil * W_fil, B * H_in * W_in)
 *
 * The grid for this kernel is launched with a 1D configuration:
 * - Grid x-dimension: covering C_in * C_out * FILTER_SIZE * FILTER_SIZE elements.
 */
template <uint FILTER_SIZE>
static __global__ void im2colInvertFilters(
        float* __restrict__ tensorOut,
        const float* __restrict__ tensorIn,
        const uint inCSize,
        const uint outCSize) {
    static_assert(FILTER_SIZE % 2);
    // Calculate filter full size and size of the whole tensor
    constexpr uint FILTER_FULL_SIZE = FILTER_SIZE * FILTER_SIZE;
    const uint tensorSize = inCSize * outCSize * FILTER_FULL_SIZE;
    // Calculate flat thread index and check bounds
    const uint tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= tensorSize) return;
    // Deconstruct the flat output index
    const uint outCHWSize = outCSize * FILTER_FULL_SIZE;
    const uint inCIdx = tIdx / outCHWSize;
    uint leftoverIdx = tIdx % outCHWSize;
    const uint outCIdx = leftoverIdx / FILTER_FULL_SIZE;
    leftoverIdx = leftoverIdx % FILTER_FULL_SIZE;
    const uint HIdx = leftoverIdx / FILTER_SIZE;
    const uint WIdx = leftoverIdx % FILTER_SIZE;
    // Calculate inverted H and W indexes
    const uint invrtHIdx = FILTER_SIZE - HIdx - 1;
    const uint invrtWIdx = FILTER_SIZE - WIdx - 1;
    // Construct the input flat index
    const uint CCIdx = outCIdx * inCSize + inCIdx;
    const uint CCIdxPart = CCIdx * FILTER_FULL_SIZE;
    const uint inHWIdx = invrtHIdx * FILTER_SIZE + invrtWIdx;
    tensorOut[tIdx] = tensorIn[CCIdxPart + inHWIdx];
}



/**
 * @brief Creates 2nd matrix intermediate im2col representation for computing dLoss/dI.
 *
 * @tparam FILTER_SIZE Size of the original convolution filter.
 * @tparam STRIDE Stride of the original convolution operation.
 *
 * @param tensorOut Output tensor in the format [C_out * FILTER_SIZE * FILTER_SIZE, B * H_in * W_in].
 * @param tensorIn Input tensor (gradient of the output) in the format [C_out, B, H_out, W_out].
 * @param CSize Number of channels.
 * @param BSize Number of batches.
 * @param inHWSize Product of the input height and width (H_in * W_in).
 * @param inWSize Width of the input tensor (W_in).
 * @param outHSize Height of the output tensor (H_out).
 * @param outWSize Width of the output tensor (W_out).
 *
 * @details
 * Performs im2col transformation preparing the second matrix for the following GEMM:
 * - (C_in, B * H_in * W_in) = (C_in, C_out * H_fil * W_fil) @ (C_out * H_fil * W_fil, B * H_in * W_in)
 * 
 * The grid for this kernel is launched with a 2D configuration:
 * - Grid x-dimension: covering B * H_in * W_in elements.
 * - Grid y-dimension: covering C_out * FILTER_SIZE * FILTER_SIZE elements.
 */
template <int FILTER_SIZE, int STRIDE>
static __global__ void im2colConvBackwardInverted(
        float* __restrict__ tensorOut,
        const float* __restrict__ tensorIn,
        const int CSize,
        const int BSize,
        const int inHWSize,
        const int inWSize,
        const int outHSize,
        const int outWSize) {
    static_assert(FILTER_SIZE % 2);
    // Calculate filter size and filter padding
    constexpr int FILTER_FULL_SIZE = FILTER_SIZE * FILTER_SIZE;
    constexpr int FILTER_PADD_SIZE = FILTER_SIZE / 2;
    // Calculate base indexes and check bounds
    const int inBHWIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const int filterCHWIdx = blockIdx.y * blockDim.y + threadIdx.y;
    const int inBHWSize = BSize * inHWSize;
    if (inBHWIdx >= inBHWSize || filterCHWIdx >= CSize * FILTER_FULL_SIZE) return;
    const int tIdx = filterCHWIdx * inBHWSize + inBHWIdx;
    // Calculate the C index and flat filter H * W index
    const int CIdx = filterCHWIdx / FILTER_FULL_SIZE;
    const int filterHWIdx = filterCHWIdx % FILTER_FULL_SIZE;
    // Calculate filter offset from convolution anchor point
    const int filterHOffsetIdx = filterHWIdx / FILTER_SIZE - FILTER_PADD_SIZE;
    const int filterWOffsetIdx = filterHWIdx % FILTER_SIZE - FILTER_PADD_SIZE;
    // Calculate the B index and flat in H * W index
    const int BIdx = inBHWIdx / inHWSize;
    const int inHWIdx = inBHWIdx % inHWSize;
    // Calculate the expanded output patch point
    const int expndPointHIdx = inHWIdx / inWSize + filterHOffsetIdx;
    const int expndPointWIdx = inHWIdx % inWSize + filterWOffsetIdx;
    // Load the patch point's value if it isn't OOB or expansion padding
    float pointValue = 0.0f;
    if (!(expndPointHIdx % STRIDE) && !(expndPointWIdx % STRIDE)) {
        // Calculate output H and W indexes for point
        const int outPointHIdx = expndPointHIdx / STRIDE;
        const int outPointWIdx = expndPointWIdx / STRIDE;
        // Check bounds for the out indexes
        if (outPointHIdx >= 0 && outPointHIdx < outHSize
                && outPointWIdx >= 0 && outPointWIdx < outWSize) {
            // Construct flat out C * B * H * W index
            const int outHWSize = outHSize * outWSize;
            const int outCIdxPart = CIdx * BSize * outHWSize;
            const int outBIdxPart = BIdx * outHWSize;
            const int outHWIdxPart = outPointHIdx * outWSize + outPointWIdx;
            // Load the corresponding point value from the out tensor
            pointValue = tensorIn[outCIdxPart + outBIdxPart + outHWIdxPart];
        }
    }
    tensorOut[tIdx] = pointValue;
}



/**
 * @brief Creates 2nd matrix intermediate im2col representation for computing depthwise dLoss/dK.
 *
 * @tparam FILTER_SIZE Size of the original convolution filter.
 * @tparam STRIDE Stride of the original convolution operation.
 *
 * @param tensorOut Output tensor in the format [C, B * H_out * W_out, FILTER_SIZE * FILTER_SIZE].
 * @param tensorIn Input tensor (gradient of the output) in the format [C, B, H_in, W_in].
 * @param outHWSize Product of the output height and width (H_out * W_out).
 * @param outWSize Width of the output tensor (W_out).
 * @param outBHWSize Product of the batch size, output height, and output width (B * H_out * W_out).
 * @param inHSize Height of the input tensor (H_in).
 * @param inWSize Width of the input tensor (W_in).
 *
 * @details
 * Performs im2col transformation preparing the second matrix for the following GEMM:
 * - (C, 1, H_fil * W_fil) = (C, 1, B * H_out * W_out) @ (C, B * H_out * W_out, H_fil * W_fil)
 *
 * The grid for this kernel is launched with a 2D configuration:
 * - Grid x-dimension: covering FILTER_SIZE * FILTER_SIZE elements.
 * - Grid y-dimension: covering C * B * H_out * W_out elements.
 */
template <int FILTER_SIZE, int STRIDE>
static __global__ void im2colConvDepthwiseBackward(
        float* __restrict__ tensorOut,
        const float* __restrict__ tensorIn,
        const int outHWSize,
        const int outWSize,
        const int outCBHWSize,
        const int inHSize,
        const int inWSize) {
    static_assert(FILTER_SIZE % 2);
    // Calculate filter size and filter padding
    constexpr int FILTER_FULL_SIZE = FILTER_SIZE * FILTER_SIZE;
    constexpr int FILTER_PADD_SIZE = FILTER_SIZE / 2;
    // Calculate base indexes and check bounds
    const int filterHWIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const int outCBHWIdx = blockIdx.y * blockDim.y + threadIdx.y;
    if (filterHWIdx >= FILTER_FULL_SIZE || outCBHWIdx >= outCBHWSize) return;
    const int tIdx = outCBHWIdx * FILTER_FULL_SIZE + filterHWIdx;
    // Calculate C * B and out H * W flat indexes
    const int CBIdx = outCBHWIdx / outHWSize;
    const int outHWIdx = outCBHWIdx % outHWSize;
    // Calculate filter offset from anchor point
    const int filterHOffsetIdx = filterHWIdx / FILTER_SIZE - FILTER_PADD_SIZE;
    const int filterWOffsetIdx = filterHWIdx % FILTER_SIZE - FILTER_PADD_SIZE;
    // Calculate the virtual patch point
    const int inPointHIdx = outHWIdx / outWSize * STRIDE + filterHOffsetIdx;
    const int inPointWIdx = outHWIdx % outWSize * STRIDE + filterWOffsetIdx;
    // Load the patch point's value if it isn't OOB
    float pointValue = 0.0f;
    if (inPointHIdx >= 0 && inPointHIdx < inHSize 
            && inPointWIdx >= 0 && inPointWIdx < inWSize) {
        const int inCBIdxPart = CBIdx * inHSize * inWSize;
        pointValue = tensorIn[inCBIdxPart + inPointHIdx * inWSize + inPointWIdx];
    }
    tensorOut[tIdx] = pointValue;
}



/**
 * @brief Performs a tiled depthwise convolution forward pass.
 *
 * @tparam BLOCK_X_SIZE Number of threads in the x-dimension of a block.
 * @tparam BLOCK_Y_SIZE Number of threads in the y-dimension of a block.
 * @tparam FILTER_SIZE Size of the square convolution filter. Must be an odd number.
 * @tparam STRIDE Stride of the convolution operation.
 *
 * @param outTensor Output tensor in the format [C, B, H_out, W_out].
 * @param inTensor Input tensor in the format [C, B, H_in, W_in].
 * @param inFilters Convolution filters in the format [C, FILTER_SIZE, FILTER_SIZE].
 * @param BSize Number of batches.
 * @param outHSize Height of the output tensor.
 * @param outWSize Width of the output tensor.
 * @param outHBlocks Number of block rows in the output height dimension.
 * @param inHSize Height of the input tensor.
 * @param inWSize Width of the input tensor.
 *
 * @details
 * The grid for this kernel is launched with a 2D configuration:
 * - Grid x-dimension: ceil(outWSize / BLOCK_X_SIZE) blocks.
 * - Grid y-dimension: CSize * BSize * outHBlocks, where outHBlocks = ceil(outHSize / BLOCK_Y_SIZE).
 */
template <
    int BLOCK_X_SIZE,
    int BLOCK_Y_SIZE,
    int FILTER_SIZE,
    int STRIDE
> static __global__
void tiledDepthwiseConvForward(
        float* __restrict__ outTensor,
        const float* __restrict__ inTensor,
        const float* __restrict__ inFilters,
        const int BSize,
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
    // Calculate W and flat C * B * H out index
    const int outWIdx = blockIdx.x * BLOCK_X_SIZE + threadIdx.x;
    const int outCBHIdx = blockIdx.y * BLOCK_Y_SIZE + threadIdx.y;
    // Calculate flat within block index
    const int tInBlockIdx = threadIdx.y * BLOCK_X_SIZE + threadIdx.x;
    // Calculate helper divisor constants
    const int outHBlockSize = outHBlocks * BLOCK_Y_SIZE;
    const int outBHBlockSize = BSize * outHBlockSize;
    // Deconstruct the flat index into B, C and H out indexes
    const int CIdx = outCBHIdx / outBHBlockSize;
    const int outLeftoverIdx = outCBHIdx % outBHBlockSize;
    const int BIdx = outLeftoverIdx / outHBlockSize;
    const int outHIdx = outLeftoverIdx % outHBlockSize;
    // Load convolution filter to the shared memory
    #pragma unroll
    for (int loadIdx = tInBlockIdx; loadIdx < FILTER_FULL_SIZE; loadIdx += THREADS_IN_BLOCK) {
        shFilter[loadIdx] = inFilters[CIdx * FILTER_FULL_SIZE + loadIdx];
    }
    __syncthreads();
    // Calculate in padding offset W and H start indexes for the block
    const int inWStartOffsetIdx = blockIdx.x * BLOCK_X_SIZE * STRIDE - FILTER_PADD_SIZE;
    const int inHStartOffsetIdx = blockIdx.y % outHBlocks * BLOCK_Y_SIZE * STRIDE - FILTER_PADD_SIZE;
    // Calculate in C * B flat index part for input indexes
    const int CBIdx = CIdx * BSize + BIdx;
    const int inCBIdxPart = CBIdx * inHSize * inWSize;
    // Load the padded tiles into shared memory
    #pragma unroll
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
        const int inLoadIdx = inCBIdxPart + inLoadHIdx * inWSize + inLoadWIdx;
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
    #pragma unroll
    for (int filterYIdx = 0; filterYIdx < FILTER_SIZE; ++filterYIdx) {
        #pragma unroll
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
    const int outCBIdxPart = CBIdx * outHSize * outWSize;
    const int tIdx = outCBIdxPart + outHIdx * outWSize + outWIdx;
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
 * @param outTensor Output tensor dLoss/dI in the format [C, B, H_in, W_in].
 * @param inTensor Non-expanded input tensor dLoss/dO in the format [C, B, H_out, W_out].
 * @param inFilters Convolution filters in the format [C, FILTER_SIZE, FILTER_SIZE].
 * @param BSize Number of batches.
 * @param outHSize Height of the output tensor.
 * @param outWSize Width of the output tensor.
 * @param outHBlocks Number of block rows in the output height dimension.
 * @param inHSize Height of the input tensor.
 * @param inWSize Width of the input tensor.
 *
 * @details
 * The grid for this kernel is launched with a 2D configuration:
 * - Grid x-dimension: ceil(outWSize / BLOCK_X_SIZE) blocks.
 * - Grid y-dimension: CSize * BSize * outHBlocks, where outHBlocks = ceil(outHSize / BLOCK_Y_SIZE).
 */
template <
    int BLOCK_X_SIZE,
    int BLOCK_Y_SIZE,
    int FILTER_SIZE,
    int STRIDE
> static __global__
void tiledDepthwiseConvBackward(
        float* __restrict__ outTensor,
        const float* __restrict__ inTensor,
        const float* __restrict__ inFilters,
        const int BSize,
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
    // Calculate W and flat C * B * H out index
    const int outWIdx = blockIdx.x * BLOCK_X_SIZE + threadIdx.x;
    const int outCBHIdx = blockIdx.y * BLOCK_Y_SIZE + threadIdx.y;
    // Calculate flat within block index
    const int tInBlockIdx = threadIdx.y * BLOCK_X_SIZE + threadIdx.x;
    // Calculate helper divisor constants
    const int outHBlockSize = outHBlocks * BLOCK_Y_SIZE;
    const int outBHBlockSize = BSize * outHBlockSize;
    // Deconstruct the flat index into B, C and H out indexes
    const int CIdx = outCBHIdx / outBHBlockSize;
    const int outLeftoverIdx = outCBHIdx % outBHBlockSize;
    const int BIdx = outLeftoverIdx / outHBlockSize;
    const int outHIdx = outLeftoverIdx % outHBlockSize;
    // Load convolution filter to the shared memory
    #pragma unroll
    for (int loadIdx = tInBlockIdx; loadIdx < FILTER_FULL_SIZE; loadIdx += THREADS_IN_BLOCK) {
        shFilter[loadIdx] = inFilters[CIdx * FILTER_FULL_SIZE + loadIdx];
    }
    __syncthreads();    
    // Calculate in padding offset W and H first needed value indexes for the block
    // FIX: Cast the arguments to 'int' to resolve template ambiguity
    const int inWFirstNeededIdx = ceilDiv((int)(blockIdx.x * BLOCK_X_SIZE - FILTER_PADD_SIZE), (int)STRIDE);
    const int inHFirstNeededIdx = ceilDiv((int)(blockIdx.y % outHBlocks * BLOCK_Y_SIZE - FILTER_PADD_SIZE), (int)STRIDE);
    // Calculate in C * B flat index part for input indexes
    const int CBIdx = CIdx * BSize + BIdx;
    const int inCBIdxPart = CBIdx * inHSize * inWSize;
    // Load the padded tiles into shared memory
    #pragma unroll
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
        const int inLoadIdx = inCBIdxPart + inLoadHIdx * inWSize + inLoadWIdx;
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
    #pragma unroll
    for (int filterYIdx = 0; filterYIdx < FILTER_SIZE; ++filterYIdx) {
        #pragma unroll
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
    const int outCBIdxPart = CBIdx * outHSize * outWSize;
    const int tIdx = outCBIdxPart + outHIdx * outWSize + outWIdx;
    outTensor[tIdx] = convolutionSum;
}
