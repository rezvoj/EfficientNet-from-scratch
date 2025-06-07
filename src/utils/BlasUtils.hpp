// src/utils/BlasUtils.hpp

#pragma once
#include <cublas_v2.h>
#include <cassert>
#include "Tensor.hpp"

// Global handle for the cuBLAS library.
inline cublasHandle_t& get_cublas_handle() {
    static cublasHandle_t handle;
    static bool is_initialized = false;
    if (!is_initialized) {
        cublasCreate(&handle);
        is_initialized = true;
    }
    return handle;
}

// C++ wrapper for GEMM: C = alpha * op(A) * op(B) + beta * C
// op(X) can be X or X^T (transpose)
inline void gemm(
    const Tensor& A,
    const Tensor& B,
    Tensor& C,
    float alpha = 1.0f,
    float beta = 0.0f,
    bool transpose_A = false,
    bool transpose_B = false) {
    
    const cublasOperation_t transa = transpose_A ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasOperation_t transb = transpose_B ? CUBLAS_OP_T : CUBLAS_OP_N;

    // Define matrix dimensions based on the operations
    const int m = transpose_A ? A.W() : A.H(); // Rows of op(A)
    const int n = transpose_B ? B.H() : B.W(); // Columns of op(B)
    const int k = transpose_A ? A.H() : A.W(); // Columns of op(A)

    // Sanity check: the "inner" dimensions must match
    const int k_b = transpose_B ? B.W() : B.H();
    assert(k == k_b);
    // Sanity check: the output dimensions must match
    assert(C.H() == m && C.W() == n);

    // FIX: The leading dimension for a row-major matrix is its width (number of columns).
    const int lda = A.W();
    const int ldb = B.W();
    const int ldc = C.W();

    // NOTE: cuBLAS assumes column-major format. To compute C=A*B for row-major
    // matrices, we can ask cuBLAS to compute C^T = B^T * A^T.
    // This is equivalent and avoids complex data reordering.
    // We achieve this by swapping A and B, swapping their transpose flags,
    // and swapping m and n.
    cublasStatus_t status = cublasSgemm(
        get_cublas_handle(),
        transb,
        transa,
        n,
        m,
        k,
        &alpha,
        B.data(),
        ldb,
        A.data(),
        lda,
        &beta,
        C.data(),
        ldc
    );
    CUBLAS_CHECK(status);
}