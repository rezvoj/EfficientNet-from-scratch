#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h> // <-- Add this include
#include <cstdint>
#include <iostream>

using uint = unsigned int;

template <typename Integer>
constexpr __host__ __device__ Integer ceilDiv(const Integer value, const Integer divisor) {
    return (value + divisor - 1) / divisor;
}

// --- For general CUDA API errors ---
void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true);
#define CUDA_CHECK(ans) gpuAssert((ans), __FILE__, __LINE__)

// --- NEW: For cuBLAS API errors ---
const char* cublasGetErrorString(cublasStatus_t status);
void cublasAssert(cublasStatus_t status, const char *file, int line, bool abort = true);
#define CUBLAS_CHECK(ans) cublasAssert((ans), __FILE__, __LINE__)