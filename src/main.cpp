// src/main.cpp

#include "utils/ImageDirectoryIterator.hpp"
#include "network/Model.hpp"
#include "network/Loss.hpp"
#include "network/Optimizer.hpp"
#include "layers/DenseLayer.hpp"
#include "layers/ActivationLayer.hpp"
#include "utils/CudaUtils.hpp"

#include <iostream>
#include <vector>

int main() {
    try {
        // --- 1. Define Training Hyperparameters ---
        const int BATCH_SIZE = 64;
        const int NUM_EPOCHS = 5;
        const float LEARNING_RATE = 1e-3;
        const cv::Size TARGET_DIM(32, 32);
        const std::string TRAIN_PATH = "resources/data/cifar10/train";
        const std::string VAL_PATH = "resources/data/cifar10/val";

        // --- 2. Build the Model ---
        Model model;
        const int input_features = TARGET_DIM.height * TARGET_DIM.width * 3;
        model.add(std::make_unique<DenseLayer>(input_features, 512));
        model.add(std::make_unique<ReLULayer>());
        model.add(std::make_unique<DenseLayer>(512, 10)); // 10 output classes for CIFAR-10
        std::cout << "Model created successfully." << std::endl;

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
            std::cout << "         EPOCH " << epoch << " / " << NUM_EPOCHS << std::endl;
            std::cout << "===============================" << std::endl;
            
            model.set_mode(true); // Set model to training mode
            ImageDirectoryIterator train_loader(TRAIN_PATH, BATCH_SIZE, TARGET_DIM, d_images, d_labels, true);
            
            int batch_count = 0;
            float total_loss = 0.0f;

            for (const auto& batch : train_loader) {
                if (batch.batch_size <= 0) continue;

                // Create a non-owning Tensor view of the batch data
                Tensor input_view(batch.images_device, batch.batch_size, 3, TARGET_DIM.height, TARGET_DIM.width);
                input_view.reshape(1, 1, batch.batch_size, input_features);

                // a. Forward pass
                Tensor predictions = model.forward(input_view);

                // b. Calculate loss
                float loss = loss_fn.forward(predictions, batch.labels_device, batch.batch_size);
                total_loss += loss * batch.batch_size;

                // c. Backward pass
                Tensor initial_grad = loss_fn.backward();
                model.backward(initial_grad);

                // d. Optimizer step
                optimizer.step();

                if (batch_count % 50 == 0) {
                    std::cout << "  Batch " << batch_count << " | Average Loss: " << total_loss / ( (batch_count * BATCH_SIZE) + batch.batch_size ) << std::endl;
                }
                batch_count++;
            }
            std::cout << "--- End of Epoch " << epoch << " | Final Average Loss: " << total_loss / 40000.0 << " ---" << std::endl;
            
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