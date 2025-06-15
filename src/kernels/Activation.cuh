#pragma once



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
