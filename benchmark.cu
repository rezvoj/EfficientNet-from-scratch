#include <cuda_runtime.h>
#include <stdio.h>   // For printf
#include <cublas_v2.h> // For cuBLAS library functions
#include <math.h>    // For ceil
#include <cuda_fp16.h> // REQUIRED for __half type

// Helper macro for CUDA error checking
#define checkCuda(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Helper macro for cuBLAS error checking
#define checkCublas(ans) { cublasAssert((ans), __FILE__, __LINE__); }
inline void cublasAssert(cublasStatus_t status, const char *file, int line, bool abort=true)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        // cuBLAS status codes are ints, not always directly map to strings like CUDA errors
        fprintf(stderr,"cuBLAS Error: %d %s %d\n", status, file, line);
        if (abort) exit(status);
    }
}

// Timing functions
__forceinline__
void cudaStartMeasuringTime(cudaEvent_t* start, cudaEvent_t* stop) {
    checkCuda(cudaEventCreate(start));
    checkCuda(cudaEventCreate(stop));
    checkCuda(cudaEventRecord(*start, 0));
}

__forceinline__
float cudaStopMeasuringTime(const cudaEvent_t start, const cudaEvent_t stop) {
    checkCuda(cudaEventRecord(stop, 0));
    checkCuda(cudaEventSynchronize(stop));
    float elapsedTime; // elapsedTime is still in float (milliseconds)
    checkCuda(cudaEventElapsedTime(&elapsedTime, start, stop));
    checkCuda(cudaEventDestroy(start));
    checkCuda(cudaEventDestroy(stop));
    return elapsedTime;
}

int main() {
    // --- Standard Convolution Parameters ---
    // (N, C_in, H_in, W_in) -> Input tensor
    // (C_out, C_in, R, S)   -> Filter tensor
    // (N, C_out, H_out, W_out) -> Output tensor

    const int N = 16;     // Batch size
    const int C_in = 32;   // Input Channels
    const int H_in = 224; // Input Height
    const int W_in = 224; // Input Width

    const int C_out = 32; // Output Channels (Number of filters)
    const int R = 3;      // Filter Height
    const int S = 3;      // Filter Width

    // Convolution parameters (for output size calculation)
    const int PADDING = 1; // Example: 1 for 'same' padding with 3x3 kernel, stride 1
    const int STRIDE = 1;
    const int DILATION = 1;

    // Calculate Output Dimensions
    // For padding=1, stride=1, dilation=1 and 3x3 kernel, H_out = H_in, W_out = W_in
    const int H_out = (H_in + 2 * PADDING - DILATION * (R - 1) - 1) / STRIDE + 1;
    const int W_out = (W_in + 2 * PADDING - DILATION * (S - 1) - 1) / STRIDE + 1;

    printf("--- Standard Convolution Parameters ---\n");
    printf("  Input: (N=%d, C_in=%d, H_in=%d, W_in=%d)\n", N, C_in, H_in, W_in);
    printf("  Filter: (C_out=%d, C_in=%d, R=%d, S=%d)\n", C_out, C_in, R, S);
    printf("  Output: (N=%d, C_out=%d, H_out=%d, W_out=%d)\n", N, C_out, H_out, W_out);
    printf("  Padding: %d, Stride: %d, Dilation: %d\n", PADDING, STRIDE, DILATION);

    // --- Derived GEMM Dimensions (using im2col concept) ---
    // The convolution (N, C_in, H_in, W_in) * (C_out, C_in, R, S) -> (N, C_out, H_out, W_out)
    // Can be mapped to a GEMM:
    // Matrix A (Filters): (C_out) x (C_in * R * S)
    // Matrix B (Im2col Input): (C_in * R * S) x (N * H_out * W_out)
    // Result C (Output): (C_out) x (N * H_out * W_out)

    const int m_gemm = C_out;                      // Rows of A and C (Output Channels)
    const int k_gemm = C_in * R * S;               // Columns of A, Rows of B (Input Channels * Filter Area)
    const int n_gemm = N * H_out * W_out;          // Columns of B and C (Batch Size * Output Spatial Area)

    printf("\n--- Equivalent cuBLAS HGEMM Dimensions (A @ B = C) ---\n");
    printf("  Matrix A (Filters, reshaped): (%d, %d)\n", m_gemm, k_gemm);
    printf("  Matrix B (Im2col Input, reshaped): (%d, %d)\n", k_gemm, n_gemm);
    printf("  Result C (Output, reshaped): (%d, %d)\n", m_gemm, n_gemm);

    // --- Device Memory Allocation ---
    __half *d_A, *d_B, *d_C_result; // Pointers for GEMM matrices, now __half

    // Allocate memory for Matrix A (Filters, original conceptual m_gemm x k_gemm)
    checkCuda(cudaMalloc((void**)&d_A, (size_t)m_gemm * k_gemm * sizeof(__half)));
    // Allocate memory for Matrix B (Im2col Input, original conceptual k_gemm x n_gemm)
    checkCuda(cudaMalloc((void**)&d_B, (size_t)k_gemm * n_gemm * sizeof(__half)));
    // Allocate memory for Matrix C (Output, original conceptual m_gemm x n_gemm)
    checkCuda(cudaMalloc((void**)&d_C_result, (size_t)m_gemm * n_gemm * sizeof(__half)));

    // Initialize device memory with dummy data (contents don't matter for performance)
    // For simplicity, using cudaMemset to zero out. For real benchmarks, you'd populate with __half values.
    checkCuda(cudaMemset(d_A, 0, (size_t)m_gemm * k_gemm * sizeof(__half)));
    checkCuda(cudaMemset(d_B, 0, (size_t)k_gemm * n_gemm * sizeof(__half)));
    checkCuda(cudaMemset(d_C_result, 0, (size_t)m_gemm * n_gemm * sizeof(__half))); // Zero out output for fresh start

    printf("\n  Allocated %.2f MB for Matrix A (Filters)\n", (double)m_gemm * k_gemm * sizeof(__half) / (1024.0 * 1024.0));
    printf("  Allocated %.2f MB for Matrix B (Im2col Input)\n", (double)k_gemm * n_gemm * sizeof(__half) / (1024.0 * 1024.0));
    printf("  Allocated %.2f MB for Matrix C (Output)\n", (double)m_gemm * n_gemm * sizeof(__half) / (1024.0 * 1024.0));

    // --- cuBLAS Handle Initialization ---
    cublasHandle_t cublasHandle;
    checkCublas(cublasCreate(&cublasHandle));

    // --- cuBLAS Hgemm Parameters ---
    // C = alpha * op(A) * op(B) + beta * C
    // Alpha and Beta must be __half for Hgemm
    const __half alpha_half = __float2half(1.0f);
    const __half beta_half = __float2half(0.0f); // Overwrite C

    cudaEvent_t start, stop;
    float elapsedTime; // Timing is still in float (ms)

    const int num_warmup_runs = 10;
    const int num_benchmark_runs = 100;

    // --- BENCHMARK: C++ cuBLAS Hgemm (Mimicking PyTorch's Internal Call) ---
    // PyTorch's C_row = A_row @ B_row is equivalent to C_col = B_row.T @ A_row.T
    //
    // For cublasHgemm(handle, transA, transB, M, N, K, alpha, A_ptr, lda, B_ptr, ldb, beta, C_ptr, ldc):
    //
    // The "A_ptr" for cuBLAS will be d_B (our original B matrix).
    // The "B_ptr" for cuBLAS will be d_A (our original A matrix).
    // The "C_ptr" for cuBLAS will be d_C_result.
    //
    // Dimensions for this cuBLAS call (M, N, K):
    // M (rows of op(A) and C) = n_gemm (rows of B.T, which is N*H_out*W_out)
    // N (cols of op(B) and C) = m_gemm (cols of A.T, which is C_out)
    // K (common dim)         = k_gemm (cols of B.T and rows of A.T, which is C_in*R*S)

    printf("\n--- Benchmarking C++ cuBLAS Hgemm (Mimicking PyTorch's Internal Call) ---\n");
    printf("  Conceptual Call: C_T(%d,%d) = B_T(%d,%d) @ A_T(%d,%d)\n",
           n_gemm, m_gemm, n_gemm, k_gemm, k_gemm, m_gemm);


    // Warm-up
    printf("  Performing warm-up runs...\n");
    for (int i = 0; i < num_warmup_runs; ++i) {
        checkCublas(cublasHgemm(cublasHandle,
                                   CUBLAS_OP_T, // transA: A_ptr (d_B) is transposed
                                   CUBLAS_OP_T, // transB: B_ptr (d_A) is transposed
                                   n_gemm,      // m: rows of op(A) and C (n_gemm from original setup)
                                   m_gemm,      // n: columns of op(B) and C (m_gemm from original setup)
                                   k_gemm,      // k: common dimension (k_gemm from original setup)
                                   &alpha_half, // Pass pointer to __half alpha
                                   d_B,         // A_ptr for cublas is d_B (original im2col matrix)
                                   k_gemm,      // lda: Leading dimension of A_ptr (d_B), which has k_gemm rows
                                   d_A,         // B_ptr for cublas is d_A (original filters matrix)
                                   m_gemm,      // ldb: Leading dimension of B_ptr (d_A), which has m_gemm rows
                                   &beta_half,  // Pass pointer to __half beta
                                   d_C_result,  // C_ptr for cublas is d_C_result (output buffer)
                                   n_gemm));    // ldc: Leading dimension of C_ptr (d_C_result), which has n_gemm rows in its transposed form
    }
    checkCuda(cudaDeviceSynchronize()); // Ensure warm-up is complete

    // Benchmark
    cudaStartMeasuringTime(&start, &stop);

    for (int i = 0; i < num_benchmark_runs; ++i) {
        checkCublas(cublasHgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T,
                                   n_gemm, m_gemm, k_gemm, &alpha_half,
                                   d_B, k_gemm, d_A, m_gemm, // Corrected lda, ldb
                                   &beta_half, d_C_result, n_gemm));
    }

    elapsedTime = cudaStopMeasuringTime(start, stop);
    printf("  Average C++ cuBLAS Hgemm (Mimicking PyTorch) time: %.4f ms\n", elapsedTime / num_benchmark_runs);


    // --- Cleanup ---
    checkCublas(cublasDestroy(cublasHandle));
    checkCuda(cudaFree(d_A));
    checkCuda(cudaFree(d_B));
    checkCuda(cudaFree(d_C_result));
    checkCuda(cudaDeviceReset()); // Resets the CUDA device, releasing all resources

    return 0;
}
