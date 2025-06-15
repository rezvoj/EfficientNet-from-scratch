#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <numeric>
#include <string>
#include "Convolution.cuh"
#include "../utils/Exceptions.cuh"
#include "../utils/Math.cuh"



template <int FILTER_SIZE, int STRIDE>
void referenceDepthwiseConv(
        float* outTensor,
        const float* inTensor,
        const float* inFilters,
        const int BSize,
        const int CSize,
        const int inHSize,
        const int inWSize,
        const int outHSize, 
        const int outWSize) {
    // Calculate padding and tensor strides
    const int padHIdx = (FILTER_SIZE - 1) / 2;
    const int padWIdx = (FILTER_SIZE - 1) / 2;
    const int inStrideBIdx = CSize * inHSize * inWSize;
    const int inStrideCIdx = inHSize * inWSize;
    const int outStrideBIdx = CSize * outHSize * outWSize;
    const int outStrideCIdx = outHSize * outWSize;
    const int filterStrideCIdx = FILTER_SIZE * FILTER_SIZE;
    // Iterate over each output element to compute its value
    for (int batchIdx = 0; batchIdx < BSize; ++batchIdx) {
        for (int channelIdx = 0; channelIdx < CSize; ++channelIdx) {
            for (int outHIdx = 0; outHIdx < outHSize; ++outHIdx) {
                for (int outWIdx = 0; outWIdx < outWSize; ++outWIdx) {
                    // Compute the convolution sum for one output element
                    float sum = 0.0f;
                    for (int filterHIdx = 0; filterHIdx < FILTER_SIZE; ++filterHIdx) {
                        for (int filterWIdx = 0; filterWIdx < FILTER_SIZE; ++filterWIdx) {
                            // Map output coordinates to input coordinates
                            const int inHIdx = outHIdx * STRIDE + filterHIdx - padHIdx;
                            const int inWIdx = outWIdx * STRIDE + filterWIdx - padWIdx;
                            // Accumulate sum, applying zero-padding implicitly
                            if (inHIdx >= 0 && inHIdx < inHSize && inWIdx >= 0 && inWIdx < inWSize) {
                                const int inTensorIdx = batchIdx * inStrideBIdx 
                                    + channelIdx * inStrideCIdx 
                                    + inHIdx * inWSize + inWIdx;
                                const int filterIdx = channelIdx * filterStrideCIdx 
                                    + filterHIdx * FILTER_SIZE + filterWIdx;
                                sum += inTensor[inTensorIdx] * inFilters[filterIdx];
                            }
                        }
                    }
                    // Store the final computed value
                    const int outTensorIdx = batchIdx * outStrideBIdx 
                        + channelIdx * outStrideCIdx 
                        + outHIdx * outWSize + outWIdx;
                    outTensor[outTensorIdx] = sum;
                }
            }
        }
    }
}



template <int FILTER_SIZE, int STRIDE>
void referenceDepthwiseConvBackward(
        float* inputGradTensor,
        const float* outputGradTensor,
        const float* filtersTensor,
        const int BSize,
        const int CSize,
        const int inHSize,
        const int inWSize,
        const int outHSize,
        const int outWSize) {
    // Calculate padding and tensor strides
    const int padHIdx = (FILTER_SIZE - 1) / 2;
    const int padWIdx = (FILTER_SIZE - 1) / 2;
    const int inputGradStrideBIdx = CSize * inHSize * inWSize;
    const int inputGradStrideCIdx = inHSize * inWSize;
    const int outputGradStrideBIdx = CSize * outHSize * outWSize;
    const int outputGradStrideCIdx = outHSize * outWSize;
    const int filterStrideCIdx = FILTER_SIZE * FILTER_SIZE;
    // Zero-out the input gradient tensor before accumulation
    for (int i = 0; i < BSize * CSize * inHSize * inWSize; ++i) {
        inputGradTensor[i] = 0.0f;
    }
    // Iterate over each output gradient to scatter its contribution
    for (int batchIdx = 0; batchIdx < BSize; ++batchIdx) {
        for (int channelIdx = 0; channelIdx < CSize; ++channelIdx) {
            for (int outHIdx = 0; outHIdx < outHSize; ++outHIdx) {
                for (int outWIdx = 0; outWIdx < outWSize; ++outWIdx) {
                    // Get the current output gradient value
                    const int outputGradIdx = batchIdx * outputGradStrideBIdx
                        + channelIdx * outputGradStrideCIdx
                        + outHIdx * outWSize + outWIdx;
                    const float outputGradVal = outputGradTensor[outputGradIdx];
                    // Apply the filter to scatter the gradient
                    for (int filterHIdx = 0; filterHIdx < FILTER_SIZE; ++filterHIdx) {
                        for (int filterWIdx = 0; filterWIdx < FILTER_SIZE; ++filterWIdx) {
                            // Map output coordinates to input coordinates
                            const int inHIdx = outHIdx * STRIDE + filterHIdx - padHIdx;
                            const int inWIdx = outWIdx * STRIDE + filterWIdx - padWIdx;
                            // Accumulate gradient, applying zero-padding implicitly
                            if (inHIdx >= 0 && inHIdx < inHSize && inWIdx >= 0 && inWIdx < inWSize) {
                                const int filterIdx = channelIdx * filterStrideCIdx
                                    + filterHIdx * FILTER_SIZE + filterWIdx;
                                const float filterVal = filtersTensor[filterIdx];
                                const int inputGradIdx = batchIdx * inputGradStrideBIdx 
                                    + channelIdx * inputGradStrideCIdx
                                    + inHIdx * inWSize + inWIdx;
                                inputGradTensor[inputGradIdx] += outputGradVal * filterVal;
                            }
                        }
                    }
                }
            }
        }
    }
}



template <int FILTER_SIZE, int STRIDE>
void referenceDepthwiseConvBackwardGrad(
        float* filterGradTensor,
        const float* outputGradTensor,
        const float* inTensor,
        const int BSize,
        const int CSize,
        const int inHSize,
        const int inWSize,
        const int outHSize,
        const int outWSize) {
    // Calculate padding and tensor strides
    const int padHIdx = (FILTER_SIZE - 1) / 2;
    const int padWIdx = (FILTER_SIZE - 1) / 2;
    const int inStrideBIdx = CSize * inHSize * inWSize;
    const int inStrideCIdx = inHSize * inWSize;
    const int outputGradStrideBIdx = CSize * outHSize * outWSize;
    const int outputGradStrideCIdx = outHSize * outWSize;
    const int filterGradStrideCIdx = FILTER_SIZE * FILTER_SIZE;
    // Zero-out the filter gradient tensor before accumulation
    for (int i = 0; i < CSize * FILTER_SIZE * FILTER_SIZE; ++i) {
        filterGradTensor[i] = 0.0f;
    }
    // Iterate over each filter weight to compute its gradient
    for (int channelIdx = 0; channelIdx < CSize; ++channelIdx) {
        for (int filterHIdx = 0; filterHIdx < FILTER_SIZE; ++filterHIdx) {
            for (int filterWIdx = 0; filterWIdx < FILTER_SIZE; ++filterWIdx) {
                // Sum the gradient contribution from all batches and positions
                float gradientSum = 0.0f;
                for (int batchIdx = 0; batchIdx < BSize; ++batchIdx) {
                    for (int outHIdx = 0; outHIdx < outHSize; ++outHIdx) {
                        for (int outWIdx = 0; outWIdx < outWSize; ++outWIdx) {
                            // Map output coordinates to input coordinates
                            const int inHIdx = outHIdx * STRIDE + filterHIdx - padHIdx;
                            const int inWIdx = outWIdx * STRIDE + filterWIdx - padWIdx;
                            // Accumulate gradient, applying zero-padding implicitly
                            if (inHIdx >= 0 && inHIdx < inHSize && inWIdx >= 0 && inWIdx < inWSize) {
                                const int inTensorIdx = batchIdx * inStrideBIdx
                                    + channelIdx * inStrideCIdx
                                    + inHIdx * inWSize + inWIdx;
                                const float inputValue = inTensor[inTensorIdx];
                                const int outputGradTensorIdx = batchIdx * outputGradStrideBIdx
                                    + channelIdx * outputGradStrideCIdx
                                    + outHIdx * outWSize + outWIdx;
                                const float outputGradValue = outputGradTensor[outputGradTensorIdx];
                                gradientSum += inputValue * outputGradValue;
                            }
                        }
                    }
                }
                // Store the final computed gradient for the filter weight
                const int filterGradTensorIdx = channelIdx * filterGradStrideCIdx
                    + filterHIdx * FILTER_SIZE + filterWIdx;
                filterGradTensor[filterGradTensorIdx] = gradientSum;
            }
        }
    }
}



struct TestParams {
    int BSize;
    int CSize;
    int inHSize;
    int inWSize;
    std::string description;
};



template <int FILTER_SIZE, int STRIDE>
bool depthwiseConvTest(const TestParams& params) {
    // Extract test parameters from input struct
    const int BSize = params.BSize;
    const int CSize = params.CSize;
    const int inHSize = params.inHSize;
    const int inWSize = params.inWSize;
    // Define CUDA block dimensions
    constexpr int BLOCK_X_SIZE = 16;
    constexpr int BLOCK_Y_SIZE = 16;
    // Print test configuration details
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Description: " << params.description << std::endl;
    std::cout << "Filter: " << FILTER_SIZE << ", Stride: " << STRIDE;
    std::cout << ", Input: [" << BSize << "," << CSize << ",";
    std::cout << inHSize << "," << inWSize << "]" << std::endl;
    // Calculate output dimensions and tensor sizes
    const int outHSize = ceilDiv(inHSize, STRIDE);
    const int outWSize = ceilDiv(inWSize, STRIDE);
    const int inTensorSize = BSize * CSize * inHSize * inWSize;
    const int outTensorSize = BSize * CSize * outHSize * outWSize;
    const int filtersSize = CSize * FILTER_SIZE * FILTER_SIZE;
    // Allocate and initialize host memory with random values
    float* h_inTensor = new float[inTensorSize];
    float* h_inFilters = new float[filtersSize];
    float* h_outTensorCPU = new float[outTensorSize];
    float* h_outTensorGPU = new float[outTensorSize];
    for (int i = 0; i < inTensorSize; ++i) h_inTensor[i] = static_cast<float>(rand() % 10);
    for (int i = 0; i < filtersSize; ++i) h_inFilters[i] = static_cast<float>(rand() % 5 - 2);
    // Allocate and copy data to device memory
    float *d_inTensor, *d_inFilters, *d_outTensor;
    checkCuda(cudaMalloc(&d_inTensor, inTensorSize * sizeof(float)));
    checkCuda(cudaMalloc(&d_inFilters, filtersSize * sizeof(float)));
    checkCuda(cudaMalloc(&d_outTensor, outTensorSize * sizeof(float)));
    checkCuda(cudaMemcpy(d_inTensor, h_inTensor, inTensorSize * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_inFilters, h_inFilters, filtersSize * sizeof(float), cudaMemcpyHostToDevice));
    // Configure CUDA kernel launch parameters
    const int outHBlocks = ceilDiv(outHSize, BLOCK_Y_SIZE);
    const dim3 gridDim(ceilDiv(outWSize, BLOCK_X_SIZE), BSize * CSize * outHBlocks);
    const dim3 blockDim(BLOCK_X_SIZE, BLOCK_Y_SIZE);
    // Launch CUDA kernel and copy back to host
    depthwiseConvForward<BLOCK_X_SIZE, BLOCK_Y_SIZE, FILTER_SIZE, STRIDE><<<gridDim, blockDim>>>(
        d_outTensor, d_inTensor, d_inFilters, 
        CSize, outHSize, outWSize, outHBlocks, 
        inHSize, inWSize
    ); 
    checkCudaLastError();
    checkCuda(cudaMemcpy(h_outTensorGPU, d_outTensor, outTensorSize * sizeof(float), cudaMemcpyDeviceToHost));
    // Run reference dethwise convolution forward implementation
    referenceDepthwiseConv<FILTER_SIZE, STRIDE>(
        h_outTensorCPU,
        h_inTensor, h_inFilters,
        BSize, CSize,
        inHSize, inWSize,
        outHSize, outWSize
    );
    // Validate results by comparing reference CPU and GPU outputs
    bool success = true;
    for (int i = 0; i < outTensorSize; ++i) {
        if (std::abs(h_outTensorCPU[i] - h_outTensorGPU[i]) > 1e-5) {
            std::cerr << "  [FAIL] Mismatch at index " << i;
            std::cerr << "! CPU: " << h_outTensorCPU[i];
            std::cerr << ", GPU: " << h_outTensorGPU[i] << std::endl;
            success = false;
            break;
        }
    }
    // Print test result and cleanup resources
    if (success) {
        std::cout << "[PASS]" << std::endl;
    }
    delete[] h_inTensor;
    delete[] h_inFilters;
    delete[] h_outTensorCPU;
    delete[] h_outTensorGPU;
    checkCuda(cudaFree(d_inTensor));
    checkCuda(cudaFree(d_inFilters));
    checkCuda(cudaFree(d_outTensor));
    return success;
}



template <int FILTER_SIZE, int STRIDE>
bool depthwiseConvBackwardTest(const TestParams& params) {
    // Extract test parameters from input struct
    const int BSize = params.BSize;
    const int CSize = params.CSize;
    const int inHSize = params.inHSize;
    const int inWSize = params.inWSize;
    // Define CUDA block dimensions
    constexpr int BLOCK_X_SIZE = 16;
    constexpr int BLOCK_Y_SIZE = 16;
    // Print test configuration details
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Description: " << params.description << std::endl;
    std::cout << "Filter: " << FILTER_SIZE << ", Stride: " << STRIDE;
    std::cout << ", Input Grad Size: [" << BSize << "," << CSize << ",";
    std::cout << inHSize << "," << inWSize << "]" << std::endl;
    // Calculate output dimensions and tensor sizes
    const int outHSize = ceilDiv(inHSize, STRIDE);
    const int outWSize = ceilDiv(inWSize, STRIDE);
    const int inputGradTensorSize = BSize * CSize * inHSize * inWSize;
    const int outputGradTensorSize = BSize * CSize * outHSize * outWSize;
    const int filtersTensorSize = CSize * FILTER_SIZE * FILTER_SIZE;
    // Allocate and initialize host memory with random values
    float* h_outputGradTensor = new float[outputGradTensorSize];
    float* h_filtersTensor = new float[filtersTensorSize];
    float* h_inputGradTensor_CPU = new float[inputGradTensorSize];
    float* h_inputGradTensor_GPU = new float[inputGradTensorSize];
    for (int i = 0; i < outputGradTensorSize; ++i) h_outputGradTensor[i] = static_cast<float>(rand() % 10 - 5);
    for (int i = 0; i < filtersTensorSize; ++i) h_filtersTensor[i] = static_cast<float>(rand() % 5 - 2);
    // Allocate and copy data to device memory
    float *d_outputGradTensor, *d_filtersTensor, *d_inputGradTensor;
    checkCuda(cudaMalloc(&d_outputGradTensor, outputGradTensorSize * sizeof(float)));
    checkCuda(cudaMalloc(&d_filtersTensor, filtersTensorSize * sizeof(float)));
    checkCuda(cudaMalloc(&d_inputGradTensor, inputGradTensorSize * sizeof(float)));
    checkCuda(cudaMemcpy(d_outputGradTensor, h_outputGradTensor, outputGradTensorSize * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_filtersTensor, h_filtersTensor, filtersTensorSize * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemset(d_inputGradTensor, 0, inputGradTensorSize * sizeof(float)));
    // Configure CUDA kernel launch parameters
    const int inHBlocks = ceilDiv(inHSize, BLOCK_Y_SIZE);
    const dim3 gridDim(ceilDiv(inWSize, BLOCK_X_SIZE), BSize * CSize * inHBlocks);
    const dim3 blockDim(BLOCK_X_SIZE, BLOCK_Y_SIZE);
    // Launch CUDA kernel and copy back to host
    depthwiseConvBackward<BLOCK_X_SIZE, BLOCK_Y_SIZE, FILTER_SIZE, STRIDE><<<gridDim, blockDim>>>(
        d_inputGradTensor, d_outputGradTensor, d_filtersTensor, 
        CSize, inHSize, inWSize, 
        inHBlocks, outHSize, outWSize
    );
    checkCudaLastError();
    checkCuda(cudaMemcpy(h_inputGradTensor_GPU, d_inputGradTensor, inputGradTensorSize * sizeof(float), cudaMemcpyDeviceToHost));
    // Run reference depthwise convolution backward implementation
    referenceDepthwiseConvBackward<FILTER_SIZE, STRIDE>(
        h_inputGradTensor_CPU, h_outputGradTensor, h_filtersTensor,
        BSize, CSize,
        inHSize, inWSize,
        outHSize, outWSize
    );
    // Validate results by comparing reference CPU and GPU outputs
    bool success = true;
    for (int i = 0; i < inputGradTensorSize; ++i) {
        if (std::abs(h_inputGradTensor_CPU[i] - h_inputGradTensor_GPU[i]) > 1e-4) {
            std::cerr << "  [FAIL] Mismatch at index " << i;
            std::cerr << "! CPU: " << h_inputGradTensor_CPU[i];
            std::cerr << ", GPU: " << h_inputGradTensor_GPU[i] << std::endl;
            success = false;
            break;
        }
    }
    // Print test result and cleanup resources
    if (success) {
        std::cout << "[PASS]" << std::endl;
    }
    delete[] h_outputGradTensor;
    delete[] h_filtersTensor;
    delete[] h_inputGradTensor_CPU;
    delete[] h_inputGradTensor_GPU;
    checkCuda(cudaFree(d_outputGradTensor));
    checkCuda(cudaFree(d_filtersTensor));
    checkCuda(cudaFree(d_inputGradTensor));
    return success;
}



template <int FILTER_SIZE, int STRIDE>
bool depthwiseConvBackwardGradTest(const TestParams& params) {
    // Extract test parameters from the input struct
    const int BSize = params.BSize;
    const int CSize = params.CSize;
    const int inHSize = params.inHSize;
    const int inWSize = params.inWSize;
    // Define a common 1D block size for the CUDA kernel
    constexpr int BLOCK_SIZE = 256;
    // Print test configuration details
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Description: " << params.description << std::endl;
    std::cout << "Filter: " << FILTER_SIZE << ", Stride: " << STRIDE;
    std::cout << ", Input Size: [" << BSize << "," << CSize << "," << inHSize << "," << inWSize << "]" << std::endl;
    // Calculate output dimensions and tensor sizes
    const int outHSize = ceilDiv(inHSize, STRIDE);
    const int outWSize = ceilDiv(inWSize, STRIDE);
    const int inTensorSize = BSize * CSize * inHSize * inWSize;
    const int outputGradTensorSize = BSize * CSize * outHSize * outWSize;
    const int filterGradTensorSize = CSize * FILTER_SIZE * FILTER_SIZE;
    // Allocate and initialize host memory with random values
    float* h_inTensor = new float[inTensorSize];
    float* h_outputGradTensor = new float[outputGradTensorSize];
    float* h_filterGradTensor_CPU = new float[filterGradTensorSize];
    float* h_filterGradTensor_GPU = new float[filterGradTensorSize];
    for (int i = 0; i < inTensorSize; ++i) h_inTensor[i] = static_cast<float>(rand() % 10 - 5) / 5.0f;
    for (int i = 0; i < outputGradTensorSize; ++i) h_outputGradTensor[i] = static_cast<float>(rand() % 10 - 5) / 5.0f;
    // Allocate and copy data to device memory
    float *d_inTensor, *d_outputGradTensor, *d_filterGradTensor;
    checkCuda(cudaMalloc(&d_inTensor, inTensorSize * sizeof(float)));
    checkCuda(cudaMalloc(&d_outputGradTensor, outputGradTensorSize * sizeof(float)));
    checkCuda(cudaMalloc(&d_filterGradTensor, filterGradTensorSize * sizeof(float)));
    checkCuda(cudaMemcpy(d_inTensor, h_inTensor, inTensorSize * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_outputGradTensor, h_outputGradTensor, outputGradTensorSize * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemset(d_filterGradTensor, 0, filterGradTensorSize * sizeof(float)));
    // Configure and launch the CUDA kernel and results from device to host
    const dim3 gridDim(CSize * FILTER_SIZE * FILTER_SIZE);
    const dim3 blockDim(BLOCK_SIZE);
    depthwiseConvBackwardGrad<BLOCK_SIZE, FILTER_SIZE, STRIDE><<<gridDim, blockDim>>>(
        d_filterGradTensor, d_outputGradTensor, d_inTensor,
        BSize, CSize, outHSize * outWSize, outWSize, 
        inHSize, inWSize
    );
    checkCudaLastError();
    checkCuda(cudaMemcpy(h_filterGradTensor_GPU, d_filterGradTensor, filterGradTensorSize * sizeof(float), cudaMemcpyDeviceToHost));
    // Run the reference implementation on the CPU to get the expected result
    referenceDepthwiseConvBackwardGrad<FILTER_SIZE, STRIDE>(
        h_filterGradTensor_CPU, h_outputGradTensor, h_inTensor,
        BSize, CSize,
        inHSize, inWSize,
        outHSize, outWSize
    );
    // Validate the results by comparing the GPU output against the CPU reference
    bool success = true;
    for (int i = 0; i < filterGradTensorSize; ++i) {
        if (std::abs(h_filterGradTensor_CPU[i] - h_filterGradTensor_GPU[i]) > 1e-3) {
            std::cerr << "  [FAIL] Mismatch at index " << i;
            std::cerr << "! CPU: " << h_filterGradTensor_CPU[i];
            std::cerr << ", GPU: " << h_filterGradTensor_GPU[i] << std::endl;
            success = false;
            break;
        }
    }
    // Print the final test result and clean up allocated resources
    if (success) {
        std::cout << "[PASS]" << std::endl;
    }
    delete[] h_inTensor;
    delete[] h_outputGradTensor;
    delete[] h_filterGradTensor_CPU;
    delete[] h_filterGradTensor_GPU;
    checkCuda(cudaFree(d_inTensor));
    checkCuda(cudaFree(d_outputGradTensor));
    checkCuda(cudaFree(d_filterGradTensor));
    return success;
}



int main() {
    // Seed the random number generator for test data and initialize counters
    srand(static_cast<unsigned int>(time(0)));
    int passed_count = 0;
    int failed_count = 0;
    // Define a set of common test parameters to be used across all test cases
    std::vector<TestParams> test_cases = {
        {2, 4, 41, 35, "Basic non-aligned dimensions"},
        {1, 1, 8, 8, "Single item, small dimensions"},
        {4, 8, 32, 64, "Dimensions potentially aligned with block sizes"},
        {3, 6, 50, 50, "Larger non-aligned dimensions"}
    };
    // Test the forward depthwise convolution
    std::cout << "=========== DEPTHWISE CONV FORWARD TESTS ===========" << std::endl;
    for (const auto& params : test_cases) {
        depthwiseConvTest<3, 1>(params) ? passed_count++ : failed_count++;
    }
    for (const auto& params : test_cases) {
        depthwiseConvTest<3, 2>(params) ? passed_count++ : failed_count++;
    }
    for (const auto& params : test_cases) {
        depthwiseConvTest<5, 1>(params) ? passed_count++ : failed_count++;
    }
    for (const auto& params : test_cases) {
        depthwiseConvTest<5, 3>(params) ? passed_count++ : failed_count++;
    }
    // Test the backward pass for the input gradient
    std::cout << "\n====== DEPTHWISE CONV BACKWARD INPUT GRADIENT TESTS ======" << std::endl;
    for (const auto& params : test_cases) {
        depthwiseConvBackwardTest<3, 1>(params) ? passed_count++ : failed_count++;
    }
    for (const auto& params : test_cases) {
        depthwiseConvBackwardTest<3, 2>(params) ? passed_count++ : failed_count++;
    }
    for (const auto& params : test_cases) {
        depthwiseConvBackwardTest<5, 1>(params) ? passed_count++ : failed_count++;
    }
    for (const auto& params : test_cases) {
        depthwiseConvBackwardTest<5, 3>(params) ? passed_count++ : failed_count++;
    }
    // Test the backward pass for the filter gradient
    std::cout << "\n===== DEPTHWISE CONV BACKWARD FILTER GRADIENT TESTS =====" << std::endl;
    for (const auto& params : test_cases) {
        depthwiseConvBackwardGradTest<3, 1>(params) ? passed_count++ : failed_count++;
    }
    for (const auto& params : test_cases) {
        depthwiseConvBackwardGradTest<3, 2>(params) ? passed_count++ : failed_count++;
    }
    for (const auto& params : test_cases) {
        depthwiseConvBackwardGradTest<5, 1>(params) ? passed_count++ : failed_count++;
    }
    for (const auto& params : test_cases) {
        depthwiseConvBackwardGradTest<5, 3>(params) ? passed_count++ : failed_count++;
    }
    // Print the final summary of all test results
    std::cout << "\n============== TEST SUMMARY ===============" << std::endl;
    std::cout << "Total test cases ran: " << (passed_count + failed_count) << std::endl;
    std::cout << "Passed: " << passed_count << std::endl;
    std::cout << "Failed: " << failed_count << std::endl;
    std::cout << "===========================================" << std::endl;
    return 0;
}
