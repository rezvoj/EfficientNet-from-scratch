
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
