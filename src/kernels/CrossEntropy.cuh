#pragma once



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
