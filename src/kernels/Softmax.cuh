
// idk maybe some fused for the expf and loss calc
struct Expf {
    __forceinline__ __device__
    float operator()(const float value) const {
        return expf(value);
    }
};
