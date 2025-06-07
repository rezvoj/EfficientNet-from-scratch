// src/main.cpp

#include "utils/ImageDirectoryIterator.hpp"
#include <iostream>
#include <cuda_runtime.h>
#include <vector>

// Helper to check for CUDA errors
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

void run_training_loop() {
    // --- 1. Define Parameters ---
    const int BATCH_SIZE = 32;
    const cv::Size TARGET_DIM(32, 32);
    const std::string TRAIN_PATH = "resources/data/cifar10/train";
    
    // --- 2. Pre-allocate GPU Memory (The Reusable Buffer Pattern) ---
    float* d_images; // "d_" for device
    int* d_labels;

    size_t image_buffer_size = 3 * BATCH_SIZE * TARGET_DIM.height * TARGET_DIM.width * sizeof(float);
    size_t label_buffer_size = BATCH_SIZE * sizeof(int);

    std::cout << "Allocating " << image_buffer_size / (1024.f * 1024.f) << " MB for images on GPU." << std::endl;
    std::cout << "Allocating " << label_buffer_size / 1024.f << " KB for labels on GPU." << std::endl;

    CUDA_CHECK(cudaMalloc(&d_images, image_buffer_size));
    CUDA_CHECK(cudaMalloc(&d_labels, label_buffer_size));

    // --- 3. Create and Use the Iterator ---
    ImageDirectoryIterator train_loader(
        TRAIN_PATH,
        BATCH_SIZE,
        TARGET_DIM,
        d_images,
        d_labels,
        true // Shuffle
    );

    int epoch = 1;
    std::cout << "\n--- Starting Epoch " << epoch << " ---" << std::endl;
    int batch_count = 0;
    for (const auto& batch : train_loader) {
        // 'batch' is a GpuBatch struct
        // The data is already on the GPU, pointed to by batch.images_device and batch.labels_device
        // We just need to know how many items are in this specific batch
        
        // This is where you would call your EfficientNet->forward() function
        // e.g., my_network.forward(batch.images_device, batch.labels_device, batch.batch_size);

        if (batch_count % 100 == 0) {
            std::cout << "  Processing Batch #" << batch_count 
                      << " (Size: " << batch.batch_size << ")" << std::endl;
        }
        batch_count++;
    }
    std::cout << "--- Epoch " << epoch << " Finished ---" << std::endl;

    // --- 4. Free GPU Memory ---
    std::cout << "\nFreeing GPU memory." << std::endl;
    CUDA_CHECK(cudaFree(d_images));
    CUDA_CHECK(cudaFree(d_labels));
}

int main() {
    try {
        run_training_loop();
    } catch (const std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}