
struct ReLU {
    __forceinline__ __device__
    float operator()(const float value) const {
        return value > 0.0f ? value : 0.0f;
    }
};


struct dReLU {
    __forceinline__ __device__
    float operator()(const float value) const {
        return value > 0.0f ? 1.0f : 0.0f;
    }
};


struct Sigmoid {
    __forceinline__ __device__
    float operator()(const float value) const {
        return 1.0f / (1.0f + expf(-value));
    }
};


struct dSigmoid {
    __forceinline__ __device__
    float operator()(const float value) const {
        const Sigmoid sigmoid;
        const float sigm = sigmoid(value);
        return sigm * (1.0f - sigm);
    }
};


struct SiLU {
    __forceinline__ __device__
    float operator()(const float value) const {
        const Sigmoid sigmoid;
        return value * sigmoid(value);
    }
};


struct dSiLU {
    __forceinline__ __device__
    float operator()(const float value) const {
        const Sigmoid sigmoid;
        const float sigm = sigmoid(value);
        return sigm * (1.0f + value * (1.0f - sigm));
    }
};
