// src/network/Optimizer.hpp

#pragma once
#include "Model.hpp"
#include <vector>

class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void step() = 0;
};

class AdamW : public Optimizer {
public:
    // --- DECLARATION ONLY ---
    AdamW(Model* model, float learning_rate = 1e-3, float beta1 = 0.9f, float beta2 = 0.999f, float weight_decay = 1e-2f, float epsilon = 1e-8f);

    void step() override;

private:
    float m_lr;
    float m_beta1;
    float m_beta2;
    float m_weight_decay;
    float m_epsilon;
    uint m_iteration;

    // A list of all trainable parameters and their gradients
    std::vector<Tensor*> m_params;
    std::vector<Tensor*> m_grads;

    // Tensors for the optimizer's internal state (m and v moments)
    std::vector<Tensor> m_m_moments;
    std::vector<Tensor> m_v_moments;
};