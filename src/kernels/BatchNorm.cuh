
static __global__
void computeMeanDiffSquared(
        float* __restrict__ outTensor,
        const float* __restrict__ inTensor,
        const float* __restrict__ sumVec,
        const uint batchSizeProd,
        const uint size) {
    const uint tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= size) return;
    const uint chanIdx = tIdx / batchSizeProd;
    const float diff = inTensor[tIdx] - sumVec[chanIdx] / batchSizeProd;
    outTensor[tIdx] = diff * diff;
}


static __global__
void computeMeansVarsUpdateRunning(
        float* __restrict__ outRunningMeans,
        float* __restrict__ outRunningVars,    
        float* __restrict__ sumsVec,
        float* __restrict__ sumsMDFVec,
        const uint batchSizeProd,
        const float momentum,
        const uint size) {
    const uint tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= size) return;
    const float batchMean = sumsVec[tIdx] / batchSizeProd;
    const float batchVar = sumsMDFVec[tIdx] / batchSizeProd;
    sumsVec[tIdx] = batchMean;
    sumsMDFVec[tIdx] = batchVar;
    outRunningMeans[tIdx] = (1 - momentum) * outRunningMeans[tIdx] + momentum * batchMean;
    outRunningVars[tIdx] = (1 - momentum) * outRunningVars[tIdx] + momentum * batchVar;
}


static __global__
void normalizeTensorInplace(
        float* __restrict__ tensor,
        const float* __restrict__ inMeansVec,
        const float* __restrict__ inVarsVec,
        const float epsilon,
        const uint batchSizeProd,
        const uint size) {
    const uint tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= size) return;
    const uint chanIdx = tIdx / batchSizeProd;
    const float mean = inMeansVec[chanIdx];
    const float var = inVarsVec[chanIdx];
    tensor[tIdx] = (tensor[tIdx] - mean) / sqrtf(var + epsilon);
}


static __global__
void scaleShiftTensor(
        float* __restrict__ outTensor,
        const float* __restrict__ inTensor,
        const float* __restrict__ inScaleVec,
        const float* __restrict__ inShiftVec,
        const uint batchSizeProd,
        const uint size) {
    const uint tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= size) return;
    const uint chanIdx = tIdx / batchSizeProd;
    const float scale = inScaleVec[chanIdx];
    const float shift = inShiftVec[chanIdx];
    outTensor[tIdx] = scale * inTensor[tIdx] + shift;
}


static __global__
void normalizeScaleShiftTensorInplace(
        float* __restrict__ tensor,
        const float* __restrict__ inMeansVec,
        const float* __restrict__ inVarsVec,
        const float* __restrict__ inScaleVec,
        const float* __restrict__ inShiftVec,
        const float epsilon,
        const uint batchSizeProd,
        const uint size) {
    const uint tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= size) return;
    const uint chanIdx = tIdx / batchSizeProd;
    const float mean = inMeansVec[chanIdx];
    const float var = inVarsVec[chanIdx];
    const float normValue = (tensor[tIdx] - mean) / sqrtf(var + epsilon);
    const float scale = inScaleVec[chanIdx];
    const float shift = inShiftVec[chanIdx];
    tensor[tIdx] = scale * normValue + shift;
}


static __global__
void batchNormPrevGradInplace(
        float* __restrict__ gradTensor,
        const float* __restrict__ inScaleVec,
        const float* __restrict__ inVarVec,
        const float* __restrict__ inNormInputTensor,
        const float* __restrict__ inShiftGradVec,
        const float* __restrict__ inScaleGradVec,
        const float epsilon,
        const uint batchSizeProd,
        const uint size) {
    const uint tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= size) return;
    const uint chanIdx = tIdx / batchSizeProd;
    const float vecPart = inScaleVec[chanIdx] / sqrtf(inVarVec[chanIdx] + epsilon);
    const float shiftGradPart = inShiftGradVec[chanIdx] / batchSizeProd;
    const float scaleGradPart = inScaleGradVec[chanIdx] / batchSizeProd * inNormInputTensor[tIdx];
    gradTensor[tIdx] = vecPart * (gradTensor[tIdx] - shiftGradPart - scaleGradPart);
}
