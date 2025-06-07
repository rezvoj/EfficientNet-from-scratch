// src/utils/ImageDirectoryIterator.cpp

#include "ImageDirectoryIterator.hpp"
#include <random>       // For std::shuffle
#include <algorithm>    // For std::min and std::shuffle
#include <iostream>
#include <chrono>       // For std::chrono

// Include the CUDA runtime API for cudaMemcpy
#include <cuda_runtime.h>

// Helper to check for CUDA errors
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

// --- ImageDirectoryIterator Class Implementation ---

ImageDirectoryIterator::ImageDirectoryIterator(
    const std::string& directory_path,
    int batch_size,
    const cv::Size& resize_dim,
    float* images_device_buffer,
    int* labels_device_buffer,
    bool shuffle)
    : m_root_path(directory_path),
      m_batch_size(batch_size),
      m_resize_dim(resize_dim),
      m_images_device_buffer(images_device_buffer),
      m_labels_device_buffer(labels_device_buffer),
      m_shuffle_on_epoch_end(shuffle) {
    
    m_cpu_image_buffer.resize(3 * m_batch_size * m_resize_dim.height * m_resize_dim.width);
    m_cpu_label_buffer.resize(m_batch_size);

    find_image_files(directory_path);
    if (m_shuffle_on_epoch_end) {
        // FIX: Use 'auto' to avoid the type conversion warning from chrono.
        auto seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::shuffle(m_file_list.begin(), m_file_list.end(), std::default_random_engine(seed));
    }
}

void ImageDirectoryIterator::find_image_files(const std::string& directory_path) {
    for (const auto& entry : fs::directory_iterator(directory_path)) {
        if (entry.is_directory()) {
            try {
                int label = std::stoi(entry.path().filename().string());
                // FIX: The main typo was here. Use "::" not "-".
                for (const auto& img_entry : fs::directory_iterator(entry.path())) {
                    if (img_entry.is_regular_file()) {
                        m_file_list.emplace_back(img_entry.path().string(), label);
                    }
                }
            } catch (const std::invalid_argument& e) {
                // Ignore directories that are not numbers
            } catch(const std::out_of_range& e) {
                // Ignore directories with names that are too large for an int
            }
        }
    }
}

ImageDirectoryIterator::iterator ImageDirectoryIterator::begin() {
    return iterator(*this, 0);
}

ImageDirectoryIterator::iterator ImageDirectoryIterator::end() {
    return iterator(*this, m_file_list.size());
}


// --- ImageDirectoryIterator::iterator Class Implementation ---

ImageDirectoryIterator::iterator::iterator(ImageDirectoryIterator& owner, size_t start_index)
    : m_owner(owner), m_current_index(start_index) {
    
    m_current_batch.images_device = m_owner.m_images_device_buffer;
    m_current_batch.labels_device = m_owner.m_labels_device_buffer;
    m_current_batch.batch_size = 0;

    if (m_current_index < m_owner.m_file_list.size()) {
        load_next_batch();
    }
}

void ImageDirectoryIterator::iterator::load_next_batch() {
    size_t remaining_files = m_owner.m_file_list.size() - m_current_index;
    
    // FIX: Cast remaining_files to int to avoid type conversion warnings in std::min
    int current_batch_size = std::min(m_owner.m_batch_size, (int)remaining_files);

    if (current_batch_size <= 0) {
        m_current_batch.batch_size = 0;
        return;
    }
    
    const int channels = 3;
    const int height = m_owner.m_resize_dim.height;
    const int width = m_owner.m_resize_dim.width;

    for (int i = 0; i < current_batch_size; ++i) {
        const auto& file_info = m_owner.m_file_list[m_current_index + i];
        
        cv::Mat img = cv::imread(file_info.first);
        if (img.empty()) {
            std::cerr << "Warning: Could not read image " << file_info.first << ". Skipping." << std::endl;
            continue;
        }

        cv::Mat resized_img;
        cv::resize(img, resized_img, m_owner.m_resize_dim);

        cv::Mat float_img;
        resized_img.convertTo(float_img, CV_32F, 1.0 / 255.0);
        
        for (int c = 0; c < channels; ++c) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    float pixel_val = float_img.at<cv::Vec3f>(h, w)[c];
                    
                    // Destination Index Calculation for [C, B, H, W] format
                    // Note: We use the *maximum* batch size for strides to ensure consistent
                    // memory layout on the GPU, even for the last smaller batch.
                    size_t dest_idx = (size_t)c * m_owner.m_batch_size * height * width +
                                      (size_t)i * height * width +
                                      (size_t)h * width +
                                      (size_t)w;

                    m_owner.m_cpu_image_buffer[dest_idx] = pixel_val;
                }
            }
        }

        m_owner.m_cpu_label_buffer[i] = file_info.second;
    }
    
    size_t image_bytes_to_copy = (size_t)channels * current_batch_size * height * width * sizeof(float);
    CUDA_CHECK(cudaMemcpy(m_owner.m_images_device_buffer, m_owner.m_cpu_image_buffer.data(), image_bytes_to_copy, cudaMemcpyHostToDevice));

    size_t label_bytes_to_copy = (size_t)current_batch_size * sizeof(int);
    CUDA_CHECK(cudaMemcpy(m_owner.m_labels_device_buffer, m_owner.m_cpu_label_buffer.data(), label_bytes_to_copy, cudaMemcpyHostToDevice));

    m_current_batch.batch_size = current_batch_size;
}


// --- Standard Iterator Boilerplate ---

const GpuBatch& ImageDirectoryIterator::iterator::operator*() const {
    return m_current_batch;
}

const GpuBatch* ImageDirectoryIterator::iterator::operator->() const {
    return &m_current_batch;
}

ImageDirectoryIterator::iterator& ImageDirectoryIterator::iterator::operator++() {
    m_current_index += m_current_batch.batch_size;
    
    if (m_current_index < m_owner.m_file_list.size()) {
        load_next_batch();
    } else {
        m_current_batch.batch_size = 0;
    }
    return *this;
}

ImageDirectoryIterator::iterator ImageDirectoryIterator::iterator::operator++(int) {
    iterator tmp = *this;
    ++(*this);
    return tmp;
}

bool ImageDirectoryIterator::iterator::operator==(const iterator& other) const {
    bool this_at_end = m_current_index >= m_owner.m_file_list.size();
    bool other_at_end = other.m_current_index >= other.m_owner.m_file_list.size();
    if (this_at_end && other_at_end) {
        return true;
    }
    return (&m_owner == &other.m_owner) && (m_current_index == other.m_current_index);
}

bool ImageDirectoryIterator::iterator::operator!=(const iterator& other) const {
    return !(*this == other);
}