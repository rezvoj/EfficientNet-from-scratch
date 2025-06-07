// tools/cifar10_converter.cpp

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <iomanip>

#include <opencv2/opencv.hpp>

// Constants remain the same
const int IMG_WIDTH = 32;
const int IMG_HEIGHT = 32;
const int IMG_CHANNELS = 3;
const int IMAGES_PER_BATCH = 10000;
const int CHANNEL_SIZE = IMG_WIDTH * IMG_HEIGHT;
const int IMAGE_BYTE_SIZE = CHANNEL_SIZE * IMG_CHANNELS;
const int RECORD_BYTE_SIZE = 1 + IMAGE_BYTE_SIZE;

namespace fs = std::filesystem;

// The process_batch_file function doesn't need any changes.
// It's generic enough to handle any batch and save it to any destination.
bool process_batch_file(const fs::path& bin_path, const fs::path& output_dir_base, int image_start_index) {
    std::cout << "Processing file: " << bin_path.string() << " -> " << output_dir_base.string() << std::endl;

    std::ifstream file(bin_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << bin_path << std::endl;
        return false;
    }

    std::vector<unsigned char> buffer(RECORD_BYTE_SIZE);

    for (int i = 0; i < IMAGES_PER_BATCH; ++i) {
        file.read(reinterpret_cast<char*>(buffer.data()), RECORD_BYTE_SIZE);

        if (file.gcount() != RECORD_BYTE_SIZE) {
            std::cerr << "Error: Failed to read full record for image " << i << std::endl;
            continue;
        }

        unsigned char label = buffer[0];
        fs::path label_dir = output_dir_base / std::to_string(label);
        fs::create_directories(label_dir);

        cv::Mat r_plane(IMG_HEIGHT, IMG_WIDTH, CV_8UC1, buffer.data() + 1);
        cv::Mat g_plane(IMG_HEIGHT, IMG_WIDTH, CV_8UC1, buffer.data() + 1 + CHANNEL_SIZE);
        cv::Mat b_plane(IMG_HEIGHT, IMG_WIDTH, CV_8UC1, buffer.data() + 1 + 2 * CHANNEL_SIZE);

        cv::Mat image_bgr;
        std::vector<cv::Mat> channels_for_merge = {b_plane, g_plane, r_plane};
        cv::merge(channels_for_merge, image_bgr);

        std::stringstream ss;
        ss << std::setw(5) << std::setfill('0') << (image_start_index + i);
        fs::path output_filepath = label_dir / (ss.str() + ".png");

        if (!cv::imwrite(output_filepath.string(), image_bgr)) {
            std::cerr << "Error: Failed to write image " << output_filepath << std::endl;
        }
    }
    
    std::cout << "  -> Finished processing " << IMAGES_PER_BATCH << " images." << std::endl;
    return true;
}

int main() {
    const fs::path data_root = "resources/data";
    const fs::path input_dir = data_root / "cifar-10-batches-bin";
    const fs::path output_dir = data_root / "cifar10";

    std::cout << "Starting CIFAR-10 to PNG conversion..." << std::endl;
    std::cout << "Input directory: " << fs::absolute(input_dir) << std::endl;
    std::cout << "Output directory: " << fs::absolute(output_dir) << std::endl;

    // --- CHANGED: Define train, validation, and test files ---
    std::vector<std::string> train_files = {
        "data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin", "data_batch_4.bin"
    };
    std::string val_file = "data_batch_5.bin"; // Using the 5th batch for validation
    std::string test_file = "test_batch.bin";

    int image_counter = 0;

    // --- Process Training Data (40,000 images) ---
    std::cout << "\n--- Processing Training Set ---" << std::endl;
    fs::path train_output_dir = output_dir / "train";
    image_counter = 0; // Start numbering from 0
    for (const auto& filename : train_files) {
        process_batch_file(input_dir / filename, train_output_dir, image_counter);
        image_counter += IMAGES_PER_BATCH;
    }

    // --- NEW: Process Validation Data (10,000 images) ---
    std::cout << "\n--- Processing Validation Set ---" << std::endl;
    fs::path val_output_dir = output_dir / "val";
    process_batch_file(input_dir / val_file, val_output_dir, 0); // Start numbering from 0

    // --- Process Test Data (10,000 images) ---
    std::cout << "\n--- Processing Test Set ---" << std::endl;
    fs::path test_output_dir = output_dir / "test";
    process_batch_file(input_dir / test_file, test_output_dir, 0); // Start numbering from 0

    std::cout << "\nConversion complete!" << std::endl;
    std::cout << "You can find the PNG images in: " << output_dir.string() << std::endl;

    return 0;
}