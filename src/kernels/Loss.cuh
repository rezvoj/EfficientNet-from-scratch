// src/kernels/Loss.cuh

#pragma once
#include "../utils/CudaUtils.hpp"

// Kernel to compute a stable softmax and the cross-entropy loss in one go.
// Input logits shape: [B, NumClasses]
// Output probs shape: [B, NumClasses]
// Labels shape: [B] (one integer label per batch item)
static __global__ void softmaxCrossEntropyFwd(
    float* __restrict__ loss,
    float* __restrict__ probs,
    const float* __restrict__ logits,
    const int* __restrict__ labels,
    int B, int NumClasses) {

    int b = blockIdx.x;
    if (b >= B) return;

    extern __shared__ float s_row[]; // Shared memory for one row of logits/probs

    // --- Step 1: Find max value in the row (done by a single thread for simplicity) ---
    if (threadIdx.x == 0) {
        float max_val = -FLT_MAX;
        for (int i = 0; i < NumClasses; ++i) {
            max_val = fmaxf(max_val, logits[b * NumClasses + i]);
        }
        s_row[NumClasses] = max_val; // Store max_val at the end of shared memory
    }
    __syncthreads(); // Ensure max_val is visible to all threads

    float max_val = s_row[NumClasses];

    // --- Step 2: Calculate exponentials and their sum ---
    float thread_sum_exp = 0.0f;
    for (int i = threadIdx.x; i < NumClasses; i += blockDim.x) {
        float val = expf(logits[b * NumClasses + i] - max_val);
        s_row[i] = val;
        thread_sum_exp += val;
    }
    __syncthreads(); // Make sure all s_row values are written

    // --- Step 3: Reduce sum, normalize, and calculate loss (done by a single thread) ---
    if (threadIdx.x == 0) {
        float total_sum_exp = 0.0f;
        for (int i = 0; i < NumClasses; ++i) {
            total_sum_exp += s_row[i];
        }

        int true_label = labels[b];
        float log_sum_exp = logf(total_sum_exp);
        float sample_loss = -(logits[b * NumClasses + true_label] - max_val - log_sum_exp);
        atomicAdd(loss, sample_loss);

        // Write out the final probabilities
        for (int i = 0; i < NumClasses; ++i) {
            probs[b * NumClasses + i] = s_row[i] / total_sum_exp;
        }
    }
}

// The backward pass for Softmax + Cross-Entropy is remarkably simple:
// dLoss/dLogit_i = (Prob_i - 1) if i is the correct class
// dLoss/dLogit_i =  Prob_i      if i is any other class
static __global__ void softmaxCrossEntropyBwd(
    float* __restrict__ d_logits, // Output: gradients w.r.t logits
    const float* __restrict__ probs,    // Input: probabilities from fwd pass
    const int* __restrict__ labels,   // Input: true labels
    int B, int NumClasses) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * NumClasses) return;

    int b = i / NumClasses; // Which batch item
    int c = i % NumClasses; // Which class logit

    int true_label = labels[b];
    
    if (c == true_label) {
        d_logits[i] = probs[i] - 1.0f;
    } else {
        d_logits[i] = probs[i];
    }

    // We also need to average the gradient over the batch size
    d_logits[i] /= B;
}