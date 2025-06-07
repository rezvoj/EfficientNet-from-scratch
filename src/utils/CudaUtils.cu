// src/utils/CudaUtils.cu

#include "CudaUtils.hpp"
#include <cstdio>
#include <cstdlib>

// Definition for general CUDA errors
void gpuAssert(cudaError_t code, const char *file, int line, bool abort) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// --- NEW: Definitions for cuBLAS errors ---

// Helper to convert cublasStatus_t to string
const char* cublasGetErrorString(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return "UNKNOWN_CUBLAS_ERROR";
}

void cublasAssert(cublasStatus_t status, const char *file, int line, bool abort) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CUBLASassert: %s %s %d\n", cublasGetErrorString(status), file, line);
        // FIX: Use the 'status' variable, not 'code'.
        if (abort) exit(status);
    }
}