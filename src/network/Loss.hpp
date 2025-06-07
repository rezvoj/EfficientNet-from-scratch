// src/network/Loss.hpp

#pragma once
#include "../utils/Tensor.hpp"

class Loss {
public:
    virtual ~Loss() = default;
    // CORRECTED SIGNATURE: Takes a raw int pointer for labels
    virtual float forward(const Tensor& predictions, const int* true_labels, int batch_size) = 0;
    virtual Tensor backward() = 0;
};

class CrossEntropyLoss : public Loss {
public:
    CrossEntropyLoss() = default;

    // CORRECTED SIGNATURE
    float forward(const Tensor& logits, const int* true_labels, int batch_size) override;
    Tensor backward() override;

private:
    Tensor m_probs_cache;      // Caches softmax probabilities
    const int* m_labels_cache; // Caches a raw pointer to the labels
    int m_batch_size_cache;    // Caches the batch size
};