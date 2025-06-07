// src/utils/ImageDirectoryIterator.hpp

#pragma once

#include <string>
#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp> // Use full header for cv::resize etc.

namespace fs = std::filesystem;

// The struct returned by the iterator.
// It contains pointers to the USER-OWNED GPU buffers and the actual size of the batch.
struct GpuBatch {
    float* images_device;
    int*   labels_device;
    int    batch_size;
};


class ImageDirectoryIterator {
public:
    class iterator;

    // --- NEW CONSTRUCTOR ---
    ImageDirectoryIterator(
        const std::string& directory_path,
        int batch_size,
        const cv::Size& resize_dim,
        float* images_device_buffer, // Pre-allocated GPU buffer for images
        int* labels_device_buffer,   // Pre-allocated GPU buffer for labels
        bool shuffle = true
    );

    iterator begin();
    iterator end();

private:
    friend class ImageDirectoryIterator::iterator; // Allow iterator access to private members

    void find_image_files(const std::string& directory_path);

    // Configuration
    std::string m_root_path;
    int m_batch_size;
    bool m_shuffle_on_epoch_end;
    cv::Size m_resize_dim;

    // Data Pointers (borrowed from the user)
    float* m_images_device_buffer;
    int* m_labels_device_buffer;

    // List of all files
    std::vector<std::pair<std::string, int>> m_file_list;

    // Pre-allocated CPU-side buffers to avoid reallocations in the loop
    std::vector<float> m_cpu_image_buffer;
    std::vector<int> m_cpu_label_buffer;
};


class ImageDirectoryIterator::iterator {
public:
    using iterator_category = std::input_iterator_tag;
    using value_type = GpuBatch; // The "real" type is GpuBatch now
    using pointer = const GpuBatch*;
    using reference = const GpuBatch&;
    using difference_type = std::ptrdiff_t;

    reference operator*() const;
    pointer operator->() const;
    iterator& operator++();
    iterator operator++(int);
    bool operator==(const iterator& other) const;
    bool operator!=(const iterator& other) const;

private:
    friend class ImageDirectoryIterator;
    iterator(ImageDirectoryIterator& owner, size_t start_index);
    void load_next_batch();

    ImageDirectoryIterator& m_owner;
    size_t m_current_index;
    GpuBatch m_current_batch; // Holds the result for the current step
};