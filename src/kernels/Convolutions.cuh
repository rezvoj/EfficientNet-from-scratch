
template <uint KERNEL_SIZE>
__global__ void im2colInvertKernel(
        float* __restrict__ tensorOut,
        const float* __restrict__ tensorIn,
        const uint chanInSize,
        const uint chanOutSize,
        const uint totalSize) {
    constexpr uint YX_SIZE_PROD = KERNEL_SIZE * KERNEL_SIZE;
    // Calculate base index and check bounds
    const uint tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= totalSize) return;
    // Deconstruct the flat output index
    const uint chanOutYXSizeProd = chanOutSize * YX_SIZE_PROD;
    const uint chanInIdx = tIdx / chanOutYXSizeProd;
    const uint leftoverIdx = tIdx % chanOutYXSizeProd;
    const uint chanOutIdx = leftoverIdx / YX_SIZE_PROD;
    leftoverIdx = leftoverIdx % YX_SIZE_PROD;
    const uint yIdx = leftoverIdx / KERNEL_SIZE;
    const uint xIdx = leftoverIdx % KERNEL_SIZE;
    // Calculate inverted y and x indexes
    const uint invrtYIdx = KERNEL_SIZE - yIdx - 1;
    const uint invrtXIdx = KERNEL_SIZE - xIdx - 1;
    // Construct the input flat index parts
    const uint chanOutchanInIdx = chanOutIdx * chanInSize + chanInIdx;
    const uint chanOutchanInIdxPart = chanOutchanInIdx * YX_SIZE_PROD;
    const uint inYXIdx = invrtYIdx * KERNEL_SIZE + invrtXIdx;
    // Copy value from input to its corresponding output cell
    tensorOut[tIdx] = tensorIn[chanOutchanInIdxPart + inYXIdx];
}


template <uint STRIDE, uint KERNEL_SIZE>
__global__ void im2colConvBackwardInverted(
        float* __restrict__ tensorOut,
        const float* __restrict__ tensorIn,
        const uint posYSize,
        const uint posXSize,
        const uint posSize,
        const uint patchSize,
        const uint inBatchSize,
        const uint inYSize,
        const uint inXSize) {

    constexpr PATCH_YX_SIZE_PROD = KERNEL_SIZE * KERNEL_SIZE;
    // Calculate base indexes and check bounds
    const uint posYXSizeProd = posYSize * posXSize;
    const uint posIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint patchIdx = blockIdx.y * blockDim.y + threadIdx.y;
    if (posIdx >= posSize || patchIdx >= patchSize) return;
    const uint tIdx = patchIdx * posSize + posIdx;

    const uint posYXIdx = posIdx % posYXSizeProd;
    const int relativePosYIdx = posYXIdx / posXSize - (posYSize - 1) / 2;
    const int relativePosXIdx = posYXIdx % posXSize - (posXSize - 1) / 2;

    const uint patchChanIdx = patchIdx / PATCH_YX_SIZE_PROD;
    const uint posBatchIdx = posIdx / posYXSizeProd;

    const uint patchYXIdx = patchIdx % PATCH_YX_SIZE_PROD;
    const uint patchYIdx = patchYXIdx / KERNEL_SIZE;
    const uint patchXIdx = patchYXIdx % KERNEL_SIZE;

    const int pointYIdx = patchYIdx + relativePosYIdx;
    const int pointXIdx = patchXIdx + relativePosXIdx;

    float pointValue = 0.0f;
    if (!(pointYIdx % STRIDE) && !(pointXIdx % STRIDE)) {
        // Calculate input y and x indexes for point
        const int inPointYIdx = pointYIdx / STRIDE;
        const int inPointXIdx = pointXIdx / STRIDE;
        // Check bounds for the input tensor
        if (inPointYIdx >= 0 && inPointYIdx < inYSize
                && inPointXIdx >= 0 && inPointXIdx < inXSize) {
            // Construct [C,B,H,W] input index
            const uint inYXSizeProd = inYSize * inXSize;
            const uint inChanIdxPart = patchChanIdx * inBatchSize * inYXSizeProd;
            const uint inBatchIdxPart = posBatchIdx * inYXSizeProd;
            const uint inYXIdxPart = inPointYIdx * inXSize + inPointXIdx;
            // Load the corresponding point value from the input tensor
            pointValue = tensorIn[inChanIdxPart + inBatchIdxPart + inYXIdxPart];
        }
    }
    tensorOut[tIdx] = pointValue
}


template <bool BACKWARD, uint STRIDE>
__global__ void im2colConv(
        float* __restrict__ tensorOut,
        const float* __restrict__ tensorIn,
        const uint posYSize,
        const uint posXSize,
        const uint posSize,
        const uint patchYXSizeProd,
        const uint patchXSize,
        const uint patchSize,
        const uint inBatchSize,
        const uint inYSize,
        const uint inXSize) {
    // Calculate base indexes and check bounds
    const uint posYXSizeProd = posYSize * posXSize;
    const uint posIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint patchIdx = blockIdx.y * blockDim.y + threadIdx.y;
    if (posIdx >= posSize || patchIdx >= patchSize) return;
    const uint tIdx = patchIdx * posSize + posIdx;

    const uint posYXIdx = posIdx % posYXSizeProd;
    const int relativePosYIdx = posYXIdx / posXSize - (posYSize - 1) / 2;
    const int relativePosXIdx = posYXIdx % posXSize - (posXSize - 1) / 2;


    uint batchIdx, chanIdx;
    if constexpr (BACKWARD) {
        batchIdx = patchIdx / patchYXSizeProd;
        chanIdx = posIdx / posYXSizeProd;
    }
    else {
        chanIdx = patchIdx / patchYXSizeProd;
        batchIdx = posIdx / posYXSizeProd;
    }

    const uint patchYXIdx = patchIdx % patchYXSizeProd;
    const uint patchYIdx = patchYXIdx / patchXSize;
    const uint patchXIdx = patchYXIdx % patchXSize;

    const uint inAnchorYIdx = patchYIdx * STRIDE;
    const uint inAnchorXIdx = patchXIdx * STRIDE;

    const int inPointYIdx = inAnchorYIdx + relativePosYIdx;
    const int inPointXIdx = inAnchorXIdx + relativePosXIdx;

    float pointValue = 0.0f;
    if (inPointYIdx >= 0 && inPointYIdx < inYSize 
            && inPointXIdx >= 0 && inPointXIdx < inXSize) {
        // Construct [C,B,H,W] input index
        const uint inYXSizeProd = inYSize * inXSize;
        const uint inChanIdxPart = chanIdx * inBatchSize * inYXSizeProd;
        const uint inBatchIdxPart = batchIdx * inYXSizeProd;
        const uint inYXIdxPart = inPointYIdx * inXSize + inPointXIdx;
        pointValue = tensorIn[inChanIdxPart + inBatchIdxPart + inYXIdxPart];
    }
    tensorOut[tIdx] = pointValue
}


template <uint STRIDE, uint KERNEL_SIZE>
__global__ void im2colConvDepthwiseBackward(
        float* __restrict__ tensorOut,
        const float* __restrict__ tensorIn,
        const uint patchYXSizeProd,
        const uint patchXSize,
        const uint patchSize,
        const uint inYSize,
        const uint inXSize) {
    // Calculate base indexes and check bounds
    const uint POS_SIZE = KERNEL_SIZE * KERNEL_SIZE;
    
    const uint posIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint patchIdx = blockIdx.y * blockDim.y + threadIdx.y;
    if (posIdx >= POS_SIZE || patchIdx >= patchSize) return;
    const uint tIdx = patchIdx * POS_SIZE + posIdx;

    const int relativePosYIdx = posIdx / KERNEL_SIZE - (KERNEL_SIZE - 1) / 2;
    const int relativePosXIdx = posIdx % KERNEL_SIZE - (KERNEL_SIZE - 1) / 2;

    const uint patchChanBatchIdx = patchIdx / patchYXSizeProd;
    const uint patchYXIdx = patchIdx % patchYXSizeProd;
    
    const uint patchYIdx = patchYXIdx / patchXSize;
    const uint patchXIdx = patchYXIdx % patchXSize;

    const uint inAnchorYIdx = patchYIdx * STRIDE;
    const uint inAnchorXIdx = patchXIdx * STRIDE;

    const int inPointYIdx = inAnchorYIdx + relativePosYIdx;
    const int inPointXIdx = inAnchorXIdx + relativePosXIdx;

    float pointValue = 0.0f;
    if (inPointYIdx >= 0 && inPointYIdx < inYSize 
            && inPointXIdx >= 0 && inPointXIdx < inXSize) {
        const uint inChanBatchIdxPart = patchChanBatchIdx * inYSize * inXSize;
        pointValue = tensorIn[inChanBatchIdxPart + inPointYIdx * inXSize + inPointXIdx];
    }
    tensorOut[tIdx] = pointValue
}


template <
    uint BLOCK_X_SIZE,
    uint BLOCK_Y_SIZE,
    uint KERNEL_SIZE,
    uint STRIDE
> __global__
void tiledDepthwiseConvBackward(
        float* __restrict__ outTensor,
        const float* __restrict__ inTensor,
        const float* __restrict__ inKernels,
        const uint dimZSize,
        const uint outDimWZYSizeProd,
        const uint outDimYSize,
        const uint outDimXSize,
        const uint inDimYSize,
        const uint inDimXSize) {
    static_assert(KERNEL_SIZE % 2);
    // Calculate template inferred constants
    constexpr threadsInBlock = BLOCK_Y_SIZE * BLOCK_X_SIZE;
    // Calculate the maximal dimension y and x tile size 
    constexpr uint expTileDimYSize = BLOCK_Y_SIZE + KERNEL_SIZE - 1;
    constexpr uint tileDimYSize = (expTileDimYSize + STRIDE - 1) / STRIDE;
    constexpr uint expTileDimXSize = BLOCK_X_SIZE + KERNEL_SIZE - 1;
    constexpr uint tileDimXSize = (expTileDimXSize + STRIDE - 1) / STRIDE;
    constexpr uint tileFullSize = tileDimYSize * tileDimXSize;
    // Calculate kernel size and kernel padding
    constexpr uint kernelFullSize = KERNEL_SIZE * KERNEL_SIZE;
    constexpr int kernelPaddSize = (KERNEL_SIZE / 2);
    // Define shared memory for kernel an tile
    __shared__ float shTile[tileFullSize];
    __shared__ float shKernel[kernelFullSize];
    // Calculate dimension x index and flat w,z,y index
    const uint outDimXBlockFirstIdx = blockIdx.x * blockDim.x;
    const uint outDimXIdx = outDimXBlockFirstIdx + threadIdx.x;
    const uint outDimWZYBlockFirstIdx = blockIdx.y * blockDim.y;
    const uint outDimWZYIdx = outDimWZYBlockFirstIdx + threadIdx.y;
    // Check bounds for the calculated indexes
    if (outDimXIdx >= outDimXSize || outDimWZYIdx >= outDimWZYSizeProd) return;
    // Calculate index within the block and flat output index
    const uint tInBlockIdx = threadIdx.y * BLOCK_X_SIZE + threadIdx.x;
    const uint tIdx = outDimWZYIdx * outDimXSize + outDimXIdx;
    // Calculate index index for the convoluted channel
    const uint dimWIdx = tIdx / (dimZSize * outDimYSize * outDimXSize);
    // Load convolution kernel to the shared memory
    #pragma unroll
    for (uint loadIdx = tInBlockIdx; loadIdx < kernelFullSize; loadIdx += threadsInBlock) {
        shKernel[loadIdx] = inKernels[dimWIdx * kernelFullSize + loadIdx];
    }
    __syncthreads();
    // Calculate dimension x and y start input index and tile size for the block
    const int inDimXStartIdx = (outDimXBlockFirstIdx - kernelPaddSize + STRIDE - 1) / STRIDE;
    const uint outDimYBlockFirstIdx = outDimWZYBlockFirstIdx % outDimYSize;
    const int inDimYStartIdx = (outDimYBlockFirstIdx - kernelPaddSize + STRIDE - 1) / STRIDE;
    // Calculate dimension w,z index part for input indexes
    const uint inDimWZIdxPart = (outDimWZYBlockFirstIdx / outDimYSize) * inDimYSize * inDimXSize;
    // Load the max needed size padded tiles into shared memory
    #pragma unroll
    for (uint loadIdx = tInBlockIdx; loadIdx < tileFullSize; loadIdx += threadsInBlock) {
        // Calculate dimension y and x input indexes
        const int inLoadDimYIdx = inDimYStartIdx + loadIdx / tileDimXSize;
        const int inLoadDimXIdx = inDimXStartIdx + loadIdx % tileDimXSize;
        // Use zero padding if out of bounds
        if (inLoadDimYIdx < 0 || inLoadDimYIdx >= inDimYSize
                || inLoadDimXIdx < 0 || inLoadDimXIdx >= inDimXSize) {
            shTile[loadIdx] = 0.0f;
            continue;
        }
        // Calculate flat input index and load convoluted region cell to shared memory
        const uint inLoadIdx = inDimWZIdxPart + inLoadDimYIdx * inDimXSize + inLoadDimXIdx;
        shTile[loadIdx] = inTensor[inLoadIdx];
    }
    __syncthreads();
    // Calculate top left convolution indexes for the expanded input to the size of output
    const int expRegionDimYStartIdx = outDimWZYIdx % outDimYSize - kernelPaddSize;
    const int expRegionDimXStartIdx = outDimXIdx - kernelPaddSize;
    // Compute inverted convolution on the kernel and expanded input region
    float convolutionSum = 0.0f;
    #pragma unroll
    for (uint kernelYIdx = 0; kernelYIdx < KERNEL_SIZE; ++kernelYIdx) {
        #pragma unroll
        for (uint kernelXIdx = 0; kernelXIdx < KERNEL_SIZE; ++kernelXIdx) {
            // Calculate the current expanded input index offset by the kernel
            const int expCurrDimYIdx = expRegionDimYStartIdx + kernelYIdx;
            const int expCurrDimXIdx = expRegionDimXStartIdx + kernelXIdx;
            // Check whether the expanded region is occupied with value or not
            if (expCurrDimYIdx % STRIDE || expCurrDimXIdx % STRIDE) continue;      
            // Calculate where in the shared memory tile is the needed value
            const uint tileDimYIdx = expCurrDimYIdx / STRIDE - inDimYStartIdx;
            const uint tileDimXIdx = expCurrDimXIdx / STRIDE - inDimXStartIdx;
            const float regionValue = shTile[tileDimYIdx * tileDimXSize + tileDimXIdx];
            // Load the corresponding inverted kernel value
            const uint invrtKernelYIdx = (KERNEL_SIZE - kernelYIdx - 1);
            const uint invrtKernelXIdx = (KERNEL_SIZE - kernelXIdx - 1);
            const float kernelValue = shKernel[invrtKernelYIdx * KERNEL_SIZE + invrtKernelXIdx];
            // Multiply the kernel value with the corresponding region cell
            convolutionSum += regionValue * kernelValue;
        }
    }
    outTensor[tIdx] = convolutionSum;
}


template <
    uint BLOCK_X_SIZE,
    uint BLOCK_Y_SIZE,
    uint KERNEL_SIZE,
    uint STRIDE
> __global__
void tiledDepthwiseConvForward(
        float* __restrict__ outTensor,
        const float* __restrict__ inTensor,
        const float* __restrict__ inKernels,
        const uint dimZSize,
        const uint outDimWZYSizeProd,
        const uint outDimYSize,
        const uint outDimXSize,
        const uint inDimYSize,
        const uint inDimXSize) {
    static_assert(KERNEL_SIZE % 2);
    // Calculate template inferred constants
    constexpr threadsInBlock = BLOCK_Y_SIZE * BLOCK_X_SIZE;
    constexpr uint tileDimYSize = BLOCK_Y_SIZE * STRIDE + KERNEL_SIZE - STRIDE;
    constexpr uint tileDimXSize = BLOCK_X_SIZE * STRIDE + KERNEL_SIZE - STRIDE;
    constexpr uint tileFullSize = tileDimYSize * tileDimXSize;
    constexpr uint kernelFullSize = KERNEL_SIZE * KERNEL_SIZE;
    constexpr int kernelPaddSize = (KERNEL_SIZE / 2);
    // Define shared memory for kernel an tile
    __shared__ float shTile[tileFullSize];
    __shared__ float shKernel[kernelFullSize];
    // Calculate dimension product between dimension x and y sizes
    const uint outDimYXSizeProd = outDimYSize * outDimXSize;
    // Calculate dimension x index and flat diemnsions y, z and w index
    const uint outDimXIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint outDimWZYIdx = blockIdx.y * blockDim.y + threadIdx.y;
    // Check bounds for the calculated indexes
    if (outDimXIdx >= outDimXSize || outDimWZYIdx >= outDimWZYSizeProd) return;
    // Calculate index within the block and flat output index
    const uint tInBlockIdx = threadIdx.y * BLOCK_X_SIZE + threadIdx.x;
    const uint tIdx = outDimWZYIdx * outDimXSize + outDimXIdx;
    // Calculate index index for the convoluted channel
    const uint dimWIdx = tIdx / (dimZSize * outDimYXSizeProd);
    // Load convolution kernel to the shared memory
    #pragma unroll
    for (uint loadIdx = tInBlockIdx; loadIdx < kernelFullSize; loadIdx += threadsInBlock) {
        shKernel[loadIdx] = inKernels[dimWIdx * kernelFullSize + loadIdx];
    }
    __syncthreads();
    // Calculate dimension x and y start indexes for the block
    const int dimXBlockStartIdx = blockIdx.x * BLOCK_X_SIZE * STRIDE - kernelPaddSize;
    const uint outDimYIdxFirstInBlock = (blockIdx.y * blockDim.y) % outDimYSize;
    const int dimYBlockStartIdx = outDimYIdxFirstInBlock * STRIDE - kernelPaddSize;
    // Calculate dimension z and w index part for input indexes
    const uint inDimWZIdxPart = (tIdx / outDimYXSizeProd) * inDimYSize * inDimXSize;
    // Load the padded tiles into shared memory
    #pragma unroll
    for (uint loadIdx = tInBlockIdx; loadIdx < tileFullSize; loadIdx += threadsInBlock) {
        // Calculate dimension x and y input indexes
        const int inLoadDimYIdx = dimYBlockStartIdx + loadIdx / tileDimXSize;
        const int inLoadDimXIdx = dimXBlockStartIdx + loadIdx % tileDimXSize;
        // Use zero padding if out of bounds
        if (inLoadDimYIdx < 0 || inLoadDimYIdx >= inDimYSize
                || inLoadDimXIdx < 0 || inLoadDimXIdx >= inDimXSize) {
            shTile[loadIdx] = 0.0f;
            continue;
        }
        // Calculate flat input index and load convoluted region cell to shared memory
        const uint inLoadIdx = inDimWZIdxPart + inLoadDimYIdx * inDimXSize + inLoadDimXIdx;
        shTile[loadIdx] = inTensor[inLoadIdx];
    }
    __syncthreads();
    // Calculate top left convolution indexes within the block
    const uint inBlockXIdx = threadIdx.x * STRIDE;
    const uint inBlockYIdx = threadIdx.y * STRIDE;
    // Compute convolution on the kernel and region
    float convolutionSum = 0.0f;
    #pragma unroll
    for (uint kernelYIdx = 0; kernelYIdx < KERNEL_SIZE; ++kernelYIdx) {
        #pragma unroll
        for (uint kernelXIdx = 0; kernelXIdx < KERNEL_SIZE; ++kernelXIdx) {
            // Load the corresponding kernel value
            const float kernelValue = shKernel[kernelYIdx * KERNEL_SIZE + kernelXIdx];
            // Calculate index of the corresponding cell
            const uint convIdxYPart = (inBlockYIdx + kernelYIdx) * tileDimXSize;
            const uint convIdxXPart = inBlockXIdx + kernelXIdx;
            // Multiply the kernel value with the corresponding region cell
            convolutionSum += shTile[convIdxYPart + convIdxXPart] * kernelValue;
        }
    }
    outTensor[tIdx] = convolutionSum;
}
