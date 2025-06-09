// src/main.cpp

#include "network/Model.hpp"
#include "network/Loss.hpp"
#include "network/Optimizer.hpp"
#include "layers/DenseLayer.hpp"
#include "layers/ActivationLayer.hpp"
#include "layers/ConvolutionLayer.hpp"
#include "layers/PoolingLayer.hpp"
#include "layers/FlattenLayer.hpp"
#include "layers/BatchNormLayer.hpp" // Added include for BatchNorm
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
        const float LEARNING_RATE = 1e-4;
        const cv::Size TARGET_DIM(32, 32);
        const float GRAD_CLIP_THRESHOLD = 1.0f;
        const std::string TRAIN_PATH = "resources/data/cifar10/train";
        const std::string VAL_PATH = "resources/data/cifar10/test"; // Using val set for testing

        // --- 2. Build the Corrected CNN Model ---
        Model model;
        // Block 1
        model.add(std::make_unique<ConvolutionLayer<3, 1>>(3, 32));   // 3 -> 32 channels
        model.add(std::make_unique<BatchNormLayer>(32));
        model.add(std::make_unique<ReLULayer>());
        model.add(std::make_unique<ConvolutionLayer<3, 1>>(32, 32));  // 32 -> 32 channels
        model.add(std::make_unique<BatchNormLayer>(32));
        model.add(std::make_unique<ReLULayer>());
        
        // TODO: A Max-Pooling layer would be better here than Global Average Pooling.
        // For now, we'll keep the global pool.
        model.add(std::make_unique<GlobalAveragePoolLayer>());       // -> [B, 32, 1, 1]
        
        model.add(std::make_unique<FlattenLayer>());                  // -> [B, 32]
        model.add(std::make_unique<DenseLayer>(32, 10));              // -> [B, 10]
        std::cout << "Deeper CNN Model created successfully." << std::endl;

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

                Tensor input_view(batch.images_device, batch.batch_size, 3, TARGET_DIM.height, TARGET_DIM.width);
                
                // a. Forward pass
                Tensor predictions = model.forward(input_view);

                // b. Calculate loss
                float loss = loss_fn.forward(predictions, batch.labels_device, batch.batch_size);
                total_loss += loss * batch.batch_size;
                total_samples += batch.batch_size;

                 // c. Backward pass
                Tensor initial_grad = loss_fn.backward();
                model.backward(initial_grad);

                // --- NEW: Step c.5: Clip the gradients ---
                optimizer.clip_gradients(GRAD_CLIP_THRESHOLD);

                // d. Optimizer step
                optimizer.step(batch.batch_size);

                if (batch_count > 0 && batch_count % 100 == 0) {
                     printf("  Batch %-5d | Avg Loss: %f\n", batch_count, total_loss / total_samples);
                }
                batch_count++;
            }
            printf("--- End of Epoch %d | Final Average Loss: %f ---\n", epoch, total_loss / total_samples);
        }

        // --- 6. Save the Final Model ---
        model.save("cifar10_cnn.bin");

        // --- 7. Evaluate on the Test Set ---
        std::cout << "\n--- Evaluating on Test Set ---" << std::endl;
        
        Model test_model;
        // Block 1
        test_model.add(std::make_unique<ConvolutionLayer<3, 1>>(3, 32));   // 3 -> 32 channels
        test_model.add(std::make_unique<BatchNormLayer>(32));
        test_model.add(std::make_unique<ReLULayer>());
        test_model.add(std::make_unique<ConvolutionLayer<3, 1>>(32, 32));  // 32 -> 32 channels
        test_model.add(std::make_unique<BatchNormLayer>(32));
        test_model.add(std::make_unique<ReLULayer>());
        
        // TODO: A Max-Pooling layer would be better here than Global Average Pooling.
        // For now, we'll keep the global pool.
        test_model.add(std::make_unique<GlobalAveragePoolLayer>());       // -> [B, 32, 1, 1]
        
        test_model.add(std::make_unique<FlattenLayer>());                  // -> [B, 32]
        test_model.add(std::make_unique<DenseLayer>(32, 10));              // -> [B, 10]
        std::cout << "Deeper CNN Model created successfully." << std::endl;
        
        test_model.load("cifar10_cnn.bin");
        test_model.set_mode(false); // CRITICAL: Set to inference mode

        ImageDirectoryIterator test_loader(VAL_PATH, BATCH_SIZE, TARGET_DIM, d_images, d_labels, false);

        int total_correct = 0;
        int total_samples_test = 0;

        for (const auto& batch : test_loader) {
            if (batch.batch_size <= 0) continue;
            
            Tensor input_view(batch.images_device, batch.batch_size, 3, TARGET_DIM.height, TARGET_DIM.width);
            
            std::vector<int> predictions = test_model.predict(input_view);

            std::vector<int> true_labels(batch.batch_size);
            CUDA_CHECK(cudaMemcpy(true_labels.data(), batch.labels_device, batch.batch_size * sizeof(int), cudaMemcpyDeviceToHost));

            for (int i = 0; i < batch.batch_size; ++i) {
                if (predictions[i] == true_labels[i]) {
                    total_correct++;
                }
            }
            total_samples_test += batch.batch_size;
        }

        double accuracy = (double)total_correct / total_samples_test;
        printf("\nTest Accuracy: %.2f%% (%d / %d)\n", accuracy * 100.0, total_correct, total_samples_test);

        // --- 8. Free GPU Memory ---
        std::cout << "\nTesting complete. Freeing GPU memory." << std::endl;
        CUDA_CHECK(cudaFree(d_images));
        CUDA_CHECK(cudaFree(d_labels));

    } catch (const std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}