// #include <iostream>
// #include <cuda_runtime.h>
// #include <cudnn.h>
// #include <cstdlib> // For std::exit

// // A simple macro to wrap cuDNN calls and check for errors.
// // This makes the main code cleaner and ensures any cuDNN issue is reported.
// #define checkCudnn(expression) { \
//     cudnnStatus_t status = (expression); \
//     if (status != CUDNN_STATUS_SUCCESS) { \
//         std::cerr << "cuDNN Error on line " << __LINE__ << ": " \
//                   << cudnnGetErrorString(status) << std::endl; \
//         std::exit(EXIT_FAILURE); \
//     } \
// }

// // A similar macro for CUDA runtime API calls.
// #define checkCuda(expression) { \
//     cudaError_t status = (expression); \
//     if (status != cudaSuccess) { \
//         std::cerr << "CUDA Error on line " << __LINE__ << ": " \
//                   << cudaGetErrorString(status) << std::endl; \
//         std::exit(EXIT_FAILURE); \
//     } \
// }


// int main() {
//     std::cout << "--- cuDNN Dropout State Size Test ---" << std::endl;

//     // 1. Get the current CUDA device and print its name
//     int deviceId;
//     checkCuda(cudaGetDevice(&deviceId));
    
//     cudaDeviceProp props;
//     checkCuda(cudaGetDeviceProperties(&props, deviceId));
//     std::cout << "Running on GPU: " << props.name << std::endl;

//     // 2. Create a handle for the cuDNN library context.
//     // This handle is needed for all subsequent cuDNN calls.
//     cudnnHandle_t cudnnHandle;
//     std::cout << "Initializing cuDNN..." << std::endl;
//     checkCudnn(cudnnCreate(&cudnnHandle));

//     // 3. Declare a variable to hold the size of the states buffer.
//     size_t statesSizeInBytes = 0;

//     // 4. Query cuDNN for the size of the dropout states buffer.
//     // This is the core function call we are testing.
//     std::cout << "Querying for dropout states buffer size..." << std::endl;
//     checkCudnn(cudnnDropoutGetStatesSize(cudnnHandle, &statesSizeInBytes));

//     // 5. Print the result.
//     std::cout << "\n----------------------------------------" << std::endl;
//     std::cout << "Result:" << std::endl;
//     std::cout << "The size of the cuDNN dropout states buffer is: " << statesSizeInBytes << " bytes." << std::endl;
//     std::cout << "----------------------------------------\n" << std::endl;
//     std::cout << "This size is independent of the input tensor size and is typically very small." << std::endl;

//     // 6. Clean up the cuDNN handle to release its resources.
//     std::cout << "Cleaning up cuDNN resources..." << std::endl;
//     checkCudnn(cudnnDestroy(cudnnHandle));

//     std::cout << "\nTest completed successfully." << std::endl;

//     return 0;
// }




#include <iostream>
#include <vector>
#include <numeric>
#include <string>
#include <algorithm>
#include <iomanip> // For std::fixed and std::setprecision

// A simple macro to wrap CUDA calls and check for errors.
#define checkCuda(expression) { \
    cudaError_t status = (expression); \
    if (status != cudaSuccess) { \
        std::cerr << "CUDA Error on line " << __LINE__ << ": " \
                  << cudaGetErrorString(status) << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
}

// Your provided timing functions
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
    float elapsedTime;
    checkCuda(cudaEventElapsedTime(&elapsedTime, start, stop));
    checkCuda(cudaEventDestroy(start));
    checkCuda(cudaEventDestroy(stop));
    return elapsedTime;
}

/**
 * @brief Runs a benchmark for cudaMalloc and cudaFree for a given size.
 * 
 * @param size_in_bytes The size of the memory to allocate in bytes.
 * @param num_iterations The number of times to repeat the test.
 * @param test_name A descriptive name for the test run.
 */
void run_benchmark(const size_t size_in_bytes, const int num_iterations, const std::string& test_name) {
    std::cout << "--- Running Benchmark: " << test_name << " ---" << std::endl;
    std::cout << "Allocating " << (double)size_in_bytes / (1024.0 * 1024.0) << " MB for " 
              << num_iterations << " iterations." << std::endl;

    std::vector<float> timings;
    timings.reserve(num_iterations);

    // Warm-up: The first call to cudaMalloc can be exceptionally slow as it
    // initializes the CUDA context. We do one call outside the timing loop.
    void* d_warmup_ptr = nullptr;
    checkCuda(cudaMalloc(&d_warmup_ptr, 1024));
    checkCuda(cudaFree(d_warmup_ptr));
    checkCuda(cudaDeviceSynchronize()); // Ensure warm-up is complete

    for (int i = 0; i < num_iterations; ++i) {
        cudaEvent_t start, stop;
        void* d_ptr = nullptr;

        cudaStartMeasuringTime(&start, &stop);

        // The operations being timed
        checkCuda(cudaMalloc(&d_ptr, size_in_bytes));
        checkCuda(cudaFree(d_ptr));

        float elapsed_ms = cudaStopMeasuringTime(start, stop);
        timings.push_back(elapsed_ms);
    }

    // Calculate statistics
    double total_time = std::accumulate(timings.begin(), timings.end(), 0.0);
    double avg_time = total_time / num_iterations;
    double min_time = *std::min_element(timings.begin(), timings.end());
    double max_time = *std::max_element(timings.begin(), timings.end());

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Results:" << std::endl;
    std::cout << "  Average Time: " << avg_time << " ms" << std::endl;
    std::cout << "  Min Time:     " << min_time << " ms" << std::endl;
    std::cout << "  Max Time:     " << max_time << " ms" << std::endl;
    std::cout << "----------------------------------------\n" << std::endl;
}


int main() {
    // Get device info
    int deviceId;
    checkCuda(cudaGetDevice(&deviceId));
    cudaDeviceProp props;
    checkCuda(cudaGetDeviceProperties(&props, deviceId));
    std::cout << "Running on GPU: " << props.name << "\n" << std::endl;

    // --- Test 1: The large tensor you requested ---
    const size_t large_tensor_elements = 128LL * 128 * 32 * 64;
    const size_t large_tensor_size = large_tensor_elements * sizeof(float);
    run_benchmark(large_tensor_size, 100, "Large Tensor (128 MB)");

    // --- Test 2: The dropout state buffer ---
    // Using 720 KB as a representative size for modern cuDNN.
    const size_t dropout_state_size = 720 * 1024; // 720 KB
    run_benchmark(dropout_state_size, 1000, "Dropout State (720 KB)");

    return 0;
}