#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include "Kernels.cuh"
#include "Utils.cuh"

constexpr int BENCHMARK_ITERATIONS = 100;



struct BenchmarkParams {
    int BSize;
    int CSize;
    int inHSize;
    int inWSize;
    int filterSize;
    int stride;
    std::string description;
};



void printBenchmarkHeader(const BenchmarkParams& params, const std::string& kernelName) {
    std::cout << std::left << std::setw(18) << kernelName
              << std::setw(35) << params.description
              << " | B:" << std::setw(4) << params.BSize
              << " | C:" << std::setw(5) << params.CSize
              << " | H:" << std::setw(5) << params.inHSize
              << " | W:" << std::setw(5) << params.inWSize
              << " | F:" << std::setw(3) << params.filterSize
              << " | S:" << std::setw(3) << params.stride;
}



template <int FILTER_SIZE, int STRIDE>
void benchmarkForward(const BenchmarkParams& params) {
    // Extract parameters
    const int BSize = params.BSize;
    const int CSize = params.CSize;
    const int inHSize = params.inHSize;
    const int inWSize = params.inWSize;
    // Define CUDA block dimensions
    constexpr int BLOCK_X_SIZE = 16;
    constexpr int BLOCK_Y_SIZE = 16;
    // Calculate output dimensions and tensor sizes
    const int outHSize = ceilDiv(inHSize, STRIDE);
    const int outWSize = ceilDiv(inWSize, STRIDE);
    const int inTensorSize = BSize * CSize * inHSize * inWSize;
    const int outTensorSize = BSize * CSize * outHSize * outWSize;
    const int filtersSize = CSize * FILTER_SIZE * FILTER_SIZE;
    // Allocate device memory
    float *d_inTensor, *d_inFilters, *d_outTensor;
    checkCuda(cudaMalloc(&d_inTensor, inTensorSize * sizeof(float)));
    checkCuda(cudaMalloc(&d_inFilters, filtersSize * sizeof(float)));
    checkCuda(cudaMalloc(&d_outTensor, outTensorSize * sizeof(float)));
    // Configure CUDA kernel launch parameters
    const int outHBlocks = ceilDiv(outHSize, BLOCK_Y_SIZE);
    const dim3 gridDim(ceilDiv(outWSize, BLOCK_X_SIZE), BSize * CSize * outHBlocks);
    const dim3 blockDim(BLOCK_X_SIZE, BLOCK_Y_SIZE);
    // Warm-up run to avoid one-time setup costs in timing
    depthwiseConvForward<BLOCK_X_SIZE, BLOCK_Y_SIZE, FILTER_SIZE, STRIDE><<<gridDim, blockDim>>>(
        d_outTensor, d_inTensor, d_inFilters,
        CSize, outHSize, outWSize, outHBlocks, inHSize, inWSize
    );
    checkCudaLastError();
    checkCuda(cudaDeviceSynchronize());
    // Start timing
    cudaEvent_t start, stop;
    cudaStartMeasuringTime(&start, &stop);
    for (int i = 0; i < BENCHMARK_ITERATIONS; ++i) {
        depthwiseConvForward<BLOCK_X_SIZE, BLOCK_Y_SIZE, FILTER_SIZE, STRIDE><<<gridDim, blockDim>>>(
            d_outTensor, d_inTensor, d_inFilters, CSize, outHSize, outWSize, outHBlocks, inHSize, inWSize);
        checkCudaLastError();
    }
    // Stop timing and calculate average
    const float totalTime = cudaStopMeasuringTime(start, stop);
    const float avgTime = totalTime / BENCHMARK_ITERATIONS;
    // Print results
    printBenchmarkHeader(params, "Forward");
    std::cout << " | Avg Time: " << std::fixed << std::setprecision(4) << avgTime << " ms" << std::endl;
    // Cleanup
    checkCuda(cudaFree(d_inTensor));
    checkCuda(cudaFree(d_inFilters));
    checkCuda(cudaFree(d_outTensor));
}



template <int FILTER_SIZE, int STRIDE>
void benchmarkBackward(const BenchmarkParams& params) {
    // Extract parameters
    const int BSize = params.BSize;
    const int CSize = params.CSize;
    const int inHSize = params.inHSize;
    const int inWSize = params.inWSize;
    // Define CUDA block dimensions
    constexpr int BLOCK_X_SIZE = 16;
    constexpr int BLOCK_Y_SIZE = 16;
    // Calculate dimensions and tensor sizes
    const int outHSize = ceilDiv(inHSize, STRIDE);
    const int outWSize = ceilDiv(inWSize, STRIDE);
    const int dLoss_dI_size = BSize * CSize * inHSize * inWSize;
    const int dLoss_dO_size = BSize * CSize * outHSize * outWSize;
    const int filtersSize = CSize * FILTER_SIZE * FILTER_SIZE;
    // Allocate device memory
    float *d_dLoss_dO, *d_filters, *d_dLoss_dI;
    checkCuda(cudaMalloc(&d_dLoss_dO, dLoss_dO_size * sizeof(float)));
    checkCuda(cudaMalloc(&d_filters, filtersSize * sizeof(float)));
    checkCuda(cudaMalloc(&d_dLoss_dI, dLoss_dI_size * sizeof(float)));
    // Configure CUDA kernel launch parameters
    const int outHBlocks = ceilDiv(inHSize, BLOCK_Y_SIZE);
    const dim3 gridDim(ceilDiv(inWSize, BLOCK_X_SIZE), BSize * CSize * outHBlocks);
    const dim3 blockDim(BLOCK_X_SIZE, BLOCK_Y_SIZE);
    // Warm-up run
    depthwiseConvBackward<BLOCK_X_SIZE, BLOCK_Y_SIZE, FILTER_SIZE, STRIDE><<<gridDim, blockDim>>>(
        d_dLoss_dI, d_dLoss_dO, d_filters,
        CSize, inHSize, inWSize, outHBlocks, outHSize, outWSize
    );
    checkCudaLastError();
    checkCuda(cudaDeviceSynchronize());
    // Start timing
    cudaEvent_t start, stop;
    cudaStartMeasuringTime(&start, &stop);
    for (int i = 0; i < BENCHMARK_ITERATIONS; ++i) {
        depthwiseConvBackward<BLOCK_X_SIZE, BLOCK_Y_SIZE, FILTER_SIZE, STRIDE><<<gridDim, blockDim>>>(
            d_dLoss_dI, d_dLoss_dO, d_filters,
            CSize, inHSize, inWSize, outHBlocks, outHSize, outWSize
        );
        checkCudaLastError();
    }
    // Stop timing and calculate average
    const float totalTime = cudaStopMeasuringTime(start, stop);
    const float avgTime = totalTime / BENCHMARK_ITERATIONS;
    // Print results
    printBenchmarkHeader(params, "Backward (Input)");
    std::cout << " | Avg Time: " << std::fixed << std::setprecision(4) << avgTime << " ms" << std::endl;
    // Cleanup
    checkCuda(cudaFree(d_dLoss_dO));
    checkCuda(cudaFree(d_filters));
    checkCuda(cudaFree(d_dLoss_dI));
}



template <int FILTER_SIZE, int STRIDE>
void benchmarkBackwardGrad(const BenchmarkParams& params) {
    // Extract parameters
    const int BSize = params.BSize;
    const int CSize = params.CSize;
    const int inHSize = params.inHSize;
    const int inWSize = params.inWSize;
    // Define block size
    constexpr int BLOCK_SIZE = 256;
    // Calculate dimensions and tensor sizes
    const int outHSize = ceilDiv(inHSize, STRIDE);
    const int outWSize = ceilDiv(inWSize, STRIDE);
    const int inTensorSize = BSize * CSize * inHSize * inWSize;
    const int outputGradTensorSize = BSize * CSize * outHSize * outWSize;
    const int filterGradTensorSize = CSize * FILTER_SIZE * FILTER_SIZE;
    // Allocate device memory
    float *d_inTensor, *d_outputGradTensor, *d_filterGradTensor;
    checkCuda(cudaMalloc(&d_inTensor, inTensorSize * sizeof(float)));
    checkCuda(cudaMalloc(&d_outputGradTensor, outputGradTensorSize * sizeof(float)));
    checkCuda(cudaMalloc(&d_filterGradTensor, filterGradTensorSize * sizeof(float)));
    // Configure CUDA kernel launch parameters
    const dim3 gridDim(CSize * FILTER_SIZE * FILTER_SIZE);
    const dim3 blockDim(BLOCK_SIZE);
    // Warm-up run
    depthwiseConvBackwardGrad<BLOCK_SIZE, FILTER_SIZE, STRIDE><<<gridDim, blockDim>>>(
        d_filterGradTensor, d_outputGradTensor, d_inTensor,
        BSize, CSize, outHSize * outWSize, outWSize, inHSize, inWSize
    );
    checkCudaLastError();
    checkCuda(cudaDeviceSynchronize());
    // Start timing
    cudaEvent_t start, stop;
    cudaStartMeasuringTime(&start, &stop);
    for (int i = 0; i < BENCHMARK_ITERATIONS; ++i) {
        depthwiseConvBackwardGrad<BLOCK_SIZE, FILTER_SIZE, STRIDE><<<gridDim, blockDim>>>(
            d_filterGradTensor, d_outputGradTensor, d_inTensor, 
            BSize, CSize, outHSize * outWSize, outWSize, inHSize, inWSize
        );
        checkCudaLastError();
    }
    // Stop timing and calculate average
    const float totalTime = cudaStopMeasuringTime(start, stop);
    const float avgTime = totalTime / BENCHMARK_ITERATIONS;
    // Print results
    printBenchmarkHeader(params, "Backward (Filter)");
    std::cout << " | Avg Time: " << std::fixed << std::setprecision(4) << avgTime << " ms" << std::endl;
    // Cleanup
    checkCuda(cudaFree(d_inTensor));
    checkCuda(cudaFree(d_outputGradTensor));
    checkCuda(cudaFree(d_filterGradTensor));
}



void runBenchmarks(const BenchmarkParams& params) {
    if (params.filterSize == 3 && params.stride == 1) {
        benchmarkForward<3, 1>(params);
        benchmarkBackward<3, 1>(params);
        benchmarkBackwardGrad<3, 1>(params);
    } 
    else if (params.filterSize == 3 && params.stride == 2) {
        benchmarkForward<3, 2>(params);
        benchmarkBackward<3, 2>(params);
        benchmarkBackwardGrad<3, 2>(params);
    } 
    else if (params.filterSize == 5 && params.stride == 1) {
        benchmarkForward<5, 1>(params);
        benchmarkBackward<5, 1>(params);
        benchmarkBackwardGrad<5, 1>(params);
    } 
    else if (params.filterSize == 5 && params.stride == 2) {
        benchmarkForward<5, 2>(params);
        benchmarkBackward<5, 2>(params);
        benchmarkBackwardGrad<5, 2>(params);
    }
    else {
        std::cerr << "Unsupported filter/stride combination for benchmark: F="
            << params.filterSize << ", S=" << params.stride << std::endl;
    }
    std::cout << "-----------------------------------------------------------------";
    std::cout << "-----------------------------------------------------------------" << std::endl;
}



int main() {
    // Define a set of benchmark cases
    std::vector<BenchmarkParams> benchmark_cases = {
        {16, 32, 224, 224, 3, 1, "Base Case (224x224, F3, S1)"},
        {16, 32, 224, 224, 3, 2, "Base Case (224x224, F3, S2)"},
        {16, 32, 224, 224, 5, 1, "Base Case (224x224, F5, S1)"},
        {16, 32, 224, 224, 5, 2, "Base Case (224x224, F5, S2)"},
        {32, 32, 224, 224, 3, 1, "Larger Batch (B32)"},
        {64, 32, 224, 224, 3, 1, "Largest Batch (B64)"},
        {16, 64, 112, 112, 3, 1, "More Channels (C64, 112x112)"},
        {16, 128, 112, 112, 3, 1, "Most Channels (C128, 112x112)"},  
        {16, 32, 112, 112, 3, 2, "Lower Resolution (112x112, S2)"},
        {32, 64, 56, 56, 3, 1, "Mixed High Load (B32, C64, 56x56)"},
        {32, 64, 56, 56, 3, 2, "Mixed High Load (B32, C64, 56x56, S2)"},
    };
    // print the benchmarking header
    std::cout << "=============================================";
    std::cout << " DEPTHWISE CONVOLUTION KERNEL BENCHMARKS ";
    std::cout << "============================================" << std::endl;
    std::cout << std::left << std::setw(18) << "Kernel"
              << std::setw(35) << "Description"
              << " | " << "Parameters" << std::endl;
    std::cout << "===========================================";
    std::cout << "===========================================";
    std::cout << "============================================" << std::endl;
    // Run all defined benchmarks
    for (const auto& params : benchmark_cases) {
        try { runBenchmarks(params); } 
        catch (const std::exception& e) {
            std::cerr << "An error occurred during benchmark: " << e.what() << std::endl;
        }
    }
    return 0;
}
