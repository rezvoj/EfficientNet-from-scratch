// src/layers/ActivationLayer.hpp

#pragma once
#include "Layer.hpp"

// We only need to declare the template here.
// The implementation will be in the .cu file.
template <typename ForwardOp, typename BackwardOp>
class ActivationLayer : public Layer {
public:
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& output_gradient) override;

private:
    Tensor m_output_cache; 
};


// We still need the forward declarations for the structs
// that will be used in the explicit instantiations.
struct ReLU;
struct dReLU;
struct SiLU;
struct dSiLU;

// We also declare our specific layer types here so other files can use them.
using ReLULayer = ActivationLayer<ReLU, dReLU>;
using SiLULayer = ActivationLayer<SiLU, dSiLU>;