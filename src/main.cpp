// src/main.cpp

#include "network/Model.hpp"
#include "network/Loss.hpp"
#include "network/Optimizer.hpp"
#include "layers/DenseLayer.hpp"
#include "layers/ActivationLayer.hpp"
#include "layers/ConvolutionLayer.hpp"
#include "layers/PoolingLayer.hpp"
#include "utils/ImageDirectoryIterator.hpp"
#include "utils/CudaUtils.hpp"

#include <iostream>
#include <vector>
#include <cstdio> // For printf

int main() {
    try {
        // --- 1. Define Training Hyperparameters ---
        const int BATCH_SIZE = 64;
        const int NUM_EPOCHS = 5;
        const float LEARNING_RATE = 1e-3;
        const cv::Size TARGET_DIM(32, 32);
        const std::string TRAIN_PATH = "resources/data/cifar10/train";
        const std::string VAL_PATH = "resources/data/cifar10/val";

        // --- 2. Build a simple CNN Model ---
        Model model;
        // Input: [B, 3, 32, 32]
        // Note: Our convolution is 'valid', so it reduces spatial dimensions.
        // H_out = (H_in - FilterSize) / Stride + 1
        model.add(std::make_unique<ConvolutionLayer<3, 1>>(3, 16)); // -> [B, 16, 30, 30]
        model.add(std::make_unique<ReLULayer>());
        model.add(std::make_unique<GlobalAveragePoolLayer>());      // -> [B, 16, 1, 1]
        // The output of the pool layer needs to be "flattened" before the dense layer.
        // The DenseLayer expects a 2D input [Batch, Features].
        // We will add a FlattenLayer next to handle this.
        model.add(std::make_unique<DenseLayer>(16, 10));             // -> [B, 10]
        std::cout << "CNN Model created successfully." << std::endl;

        // --- 3. Create Loss Function and Optimizer ---
        CrossEntropyLoss loss_fn;
        AdamW optimizer(&model, LEARNING_RATE);
        std::cout << "Loss function and optimizer are ready." << std::endl;

        // --- 4. Allocate GPU Buffers for Data Loader ---
        float* d_images;
        int* d_labels;
        size_t image_buffer_size = 3 * BATCH_SIZE * TARGET_DIM.height * TARGET_DIM.width * sizeof(float);
        size_t label_buffer_size = BATCH_SIZE * sizeof(int);
        CUDA_CHECK(cudaMalloc(&d_images, image_buffer_size));
        CUDA_CHECK(cudaMalloc(&d_labels, label_buffer_size));
        std::cout << "GPU data buffers allocated." << std::endl;

        // --- 5. The Training Loop ---
        for (int epoch = 1; epoch <= NUM_EPOCHS; ++epoch) {
            std::cout << "\n===============================" << std::endl;
            printf("         EPOCH %d / %d\n", epoch, NUM_EPOCHS);
            std::cout << "===============================" << std::endl;
            
            model.set_mode(true); // Set model to training mode
            ImageDirectoryIterator train_loader(TRAIN_PATH, BATCH_SIZE, TARGET_DIM, d_images, d_labels, true);
            
            int batch_count = 0;
            float total_loss = 0.0f;
            int total_samples = 0;

            for (const auto& batch : train_loader) {
                if (batch.batch_size <= 0) continue;

                // Create a non-owning Tensor view of the batch data
                Tensor input_view(batch.images_device, batch.batch_size, 3, TARGET_DIM.height, TARGET_DIM.width);
                
                // a. Forward pass
                // The output of the pooling layer is [B, 16, 1, 1]. The dense layer expects [B, 16].
                // Our current model will fail here. We need a FlattenLayer.
                // For now, we will manually reshape before the Dense layer in the model's forward pass.
                Tensor predictions = model.forward(input_view);

                // b. Calculate loss
                float loss = loss_fn.forward(predictions, batch.labels_device, batch.batch_size);
                total_loss += loss * batch.batch_size;
                total_samples += batch.batch_size;

                // c. Backward pass
                Tensor initial_grad = loss_fn.backward();
                model.backward(initial_grad);

                // d. Optimizer step
                optimizer.step();

                if (batch_count > 0 && batch_count % 100 == 0) {
                     printf("  Batch %-5d | Avg Loss: %f\n", batch_count, total_loss / total_samples);
                }
                batch_count++;
            }
            printf("--- End of Epoch %d | Final Average Loss: %f ---\n", epoch, total_loss / total_samples);
            
            // TODO: Add a validation loop here using the same pattern but with model.set_mode(false)
            // and without the backward/optimizer steps.
        }

        // --- 6. Free GPU Memory ---
        std::cout << "\nTraining complete. Freeing GPU memory." << std::endl;
        CUDA_CHECK(cudaFree(d_images));
        CUDA_CHECK(cudaFree(d_labels));

    } catch (const std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}