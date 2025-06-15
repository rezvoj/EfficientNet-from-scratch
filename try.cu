#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#define checkCuda(func) do { cudaError_t err = (func); if (err != cudaSuccess) { std::cerr << "CUDA error " << err << " at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; exit(EXIT_FAILURE); } } while (0)
#define checkCublas(func) do { cublasStatus_t status = (func); if (status != CUBLAS_STATUS_SUCCESS) { std::cerr << "cuBLAS error " << status << " at " << __FILE__ << ":" << __LINE__ << std::endl; exit(EXIT_FAILURE); } } while (0)

__device__ inline float warpReduce(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}


__global__ void reduce_cols_kernel(const float* input, float* output, int N, int C) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / 32;
    const int thread_id = threadIdx.x;
    const int warp_id = thread_id / 32;
    const int lane_id = thread_id % 32;
    const int col_idx = blockIdx.x;
    __shared__ float partial_sums[WARPS_PER_BLOCK];

    float thread_sum = 0.0f;
    for (int i = thread_id; i < N; i += BLOCK_SIZE) {
        thread_sum += input[(size_t)i * C + col_idx];
    }
    thread_sum = warpReduce(thread_sum);
    if (lane_id == 0) {
        partial_sums[warp_id] = thread_sum;
    }
    __syncthreads();
    if (warp_id == 0) {
        float final_sum = (lane_id < WARPS_PER_BLOCK) ? partial_sums[lane_id] : 0.0f;
        final_sum = warpReduce(final_sum);
        if (lane_id == 0) {
            output[col_idx] = final_sum;
        }
    }
}

int main() {
    int N = 3200;
    int C = 5120;
    int iterations = 1000;

    // --- Host Data and Ground Truth Calculation ---
    std::vector<float> h_input( (size_t)N * C, 1.0f);
    std::vector<float> h_output_cpu(C);

    for (int i = 0; i < C; ++i) {
        float col_sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            col_sum += h_input[(size_t)j * C + i];
        }
        h_output_cpu[i] = col_sum;
    }

    // --- Device Allocation and Data Transfer ---
    float *d_input, *d_ones, *d_output_kernel, *d_output_cublas;
    checkCuda(cudaMalloc(&d_input, (size_t)N * C * sizeof(float)));
    checkCuda(cudaMalloc(&d_ones, N * sizeof(float)));
    checkCuda(cudaMalloc(&d_output_kernel, C * sizeof(float)));
    checkCuda(cudaMalloc(&d_output_cublas, C * sizeof(float)));

    checkCuda(cudaMemcpy(d_input, h_input.data(), (size_t)N * C * sizeof(float), cudaMemcpyHostToDevice));
    std::vector<float> h_ones(N, 1.0f);
    checkCuda(cudaMemcpy(d_ones, h_ones.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t cublas_handle;
    checkCublas(cublasCreate(&cublas_handle));

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&stop));
    float time_ms;

    // --- Custom Kernel Execution ---
    dim3 grid_dim(C, 1, 1);
    dim3 block_dim(256, 1, 1);
    checkCuda(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        reduce_cols_kernel<<<grid_dim, block_dim>>>(d_input, d_output_kernel, N, C);
    }
    checkCuda(cudaEventRecord(stop));
    checkCuda(cudaEventSynchronize(stop));
    checkCuda(cudaEventElapsedTime(&time_ms, start, stop));
    std::cout << "Custom Kernel: " << std::fixed << std::setprecision(6) << time_ms / iterations << " ms" << std::endl;


    // --- cuBLAS Execution ---
    const float alpha = 1.0f, beta = 0.0f;
    checkCuda(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        checkCublas(cublasSgemv(cublas_handle, CUBLAS_OP_N, C, N, &alpha, d_input, C, d_ones, 1, &beta, d_output_cublas, 1));
    }
    checkCuda(cudaEventRecord(stop));
    checkCuda(cudaEventSynchronize(stop));
    checkCuda(cudaEventElapsedTime(&time_ms, start, stop));
    std::cout << "cuBLAS Sgemv:  " << std::fixed << std::setprecision(6) << time_ms / iterations << " ms" << std::endl;

    // --- Verification ---
    std::vector<float> h_output_kernel(C);
    std::vector<float> h_output_cublas(C);
    checkCuda(cudaMemcpy(h_output_kernel.data(), d_output_kernel, C * sizeof(float), cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(h_output_cublas.data(), d_output_cublas, C * sizeof(float), cudaMemcpyDeviceToHost));

    bool kernel_ok = true;
    bool cublas_ok = true;

    for (int i = 0; i < C; ++i) {
        if (std::abs(h_output_kernel[i] - h_output_cpu[i]) > 1e-3) kernel_ok = false;
        if (std::abs(h_output_cublas[i] - h_output_cpu[i]) > 1e-3) cublas_ok = false;
    }

    std::cout << "-------------------------------------------" << std::endl;
    std::cout << "CPU Ground Truth vs. GPU Results" << std::endl;
    std::cout << "-------------------------------------------" << std::endl;
    std::cout << "Custom Kernel: " << (kernel_ok ? "CORRECT" : "WRONG") << std::endl;
    std::cout << "cuBLAS Sgemv:  " << (cublas_ok ? "CORRECT" : "WRONG") << std::endl;
    std::cout << "-------------------------------------------" << std::endl;

    if (!kernel_ok || !cublas_ok) {
        for (int i = 0; i < C; ++i) {
            if (std::abs(h_output_kernel[i] - h_output_cpu[i]) > 1e-3 || std::abs(h_output_cublas[i] - h_output_cpu[i]) > 1e-3) {
                std::cout << "First mismatch at index " << i << ":" << std::endl;
                std::cout << "  - CPU Result:    " << h_output_cpu[i] << std::endl;
                std::cout << "  - Kernel Result: " << h_output_kernel[i] << std::endl;
                std::cout << "  - cuBLAS Result: " << h_output_cublas[i] << std::endl;
                break;
            }
        }
    }

    // --- Cleanup ---
    checkCuda(cudaFree(d_input));
    checkCuda(cudaFree(d_ones));
    checkCuda(cudaFree(d_output_kernel));
    checkCuda(cudaFree(d_output_cublas));
    checkCublas(cublasDestroy(cublas_handle));

    return 0;
}
