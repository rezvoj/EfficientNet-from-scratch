// src/utils/Tensor.hpp

#pragma once
#include <vector>
#include <numeric>
#include <stdexcept>
#include <cassert>
#include <iostream>
#include <algorithm> // For std::min
#include "CudaUtils.hpp"

// A class to manage a tensor on the GPU.
// It handles memory allocation/deallocation via RAII using an ownership flag.
class Tensor {
public:
    // Default constructor for an empty tensor
    Tensor() : m_ptr(nullptr), m_b(0), m_c(0), m_h(0), m_w(0), m_owns_memory(false) {}

    // Constructor to create a new, OWNING tensor of a given shape
    Tensor(int b, int c, int h, int w) : m_b(b), m_c(c), m_h(h), m_w(w), m_owns_memory(true) {
        size_t num_elements = (size_t)b * c * h * w;
        if (num_elements > 0) {
            CUDA_CHECK(cudaMalloc(&m_ptr, num_elements * sizeof(float)));
        } else {
            m_ptr = nullptr;
        }
    }

    // Constructor to create a NON-OWNING "view" of existing GPU memory
    Tensor(float* existing_ptr, int b, int c, int h, int w) 
        : m_ptr(existing_ptr), m_b(b), m_c(c), m_h(h), m_w(w), m_owns_memory(false) {
        // This tensor is just a view, it does not own the memory.
    }

    // Destructor: only frees memory if it owns it.
    ~Tensor() {
        if (m_ptr != nullptr && m_owns_memory) {
            cudaFree(m_ptr);
        }
    }

    // --- Rule of Five for proper resource management ---
    
    // 1. Copy Constructor (always creates a new, owning deep copy)
    Tensor(const Tensor& other) 
        : m_b(other.m_b), m_c(other.m_c), m_h(other.m_h), m_w(other.m_w), m_owns_memory(true) {
        size_t num_elements = other.size();
        if (num_elements > 0) {
            CUDA_CHECK(cudaMalloc(&m_ptr, num_elements * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(m_ptr, other.m_ptr, num_elements * sizeof(float), cudaMemcpyDeviceToDevice));
        } else {
            m_ptr = nullptr;
        }
    }

    // 2. Copy Assignment Operator (replaces content with a new, owning deep copy)
    Tensor& operator=(const Tensor& other) {
        if (this == &other) return *this; // Self-assignment check

        // If this tensor currently owns memory, free it before reassigning.
        if (m_ptr && m_owns_memory) {
            cudaFree(m_ptr);
        }

        // Copy data and dimensions
        m_b = other.m_b; m_c = other.m_c; m_h = other.m_h; m_w = other.m_w;
        m_owns_memory = true; // The new copy ALWAYS owns its memory.
        
        size_t num_elements = other.size();
        if (num_elements > 0) {
            CUDA_CHECK(cudaMalloc(&m_ptr, num_elements * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(m_ptr, other.m_ptr, num_elements * sizeof(float), cudaMemcpyDeviceToDevice));
        } else {
            m_ptr = nullptr;
        }
        return *this;
    }

    // 3. Move Constructor (efficiently transfers ownership)
    Tensor(Tensor&& other) noexcept 
        : m_ptr(other.m_ptr), m_b(other.m_b), m_c(other.m_c), m_h(other.m_h), m_w(other.m_w), m_owns_memory(other.m_owns_memory) {
        // Leave the moved-from object in a valid, non-owning, empty state
        other.m_ptr = nullptr;
        other.m_owns_memory = false;
        other.m_b = 0; other.m_c = 0; other.m_h = 0; other.m_w = 0;
    }

    // 4. Move Assignment Operator (efficiently transfers ownership)
    Tensor& operator=(Tensor&& other) noexcept {
        if (this == &other) return *this;
        if (m_ptr && m_owns_memory) cudaFree(m_ptr);

        m_ptr = other.m_ptr;
        m_owns_memory = other.m_owns_memory;
        m_b = other.m_b; m_c = other.m_c; m_h = other.m_h; m_w = other.m_w;

        other.m_ptr = nullptr;
        other.m_owns_memory = false;
        other.m_b = 0; other.m_c = 0; other.m_h = 0; other.m_w = 0;
        return *this;
    }
    
    // --- Accessors ---
    float* data() const { return m_ptr; }
    int B() const { return m_b; }
    int C() const { return m_c; }
    int H() const { return m_h; }
    int W() const { return m_w; }
    size_t size() const { return (size_t)m_b * m_c * m_h * m_w; }

    // --- Mutators ---
    void reshape(int b, int c, int h, int w) {
        // Reshape is only allowed if the total number of elements remains the same.
        assert((size_t)b * c * h * w == this->size());
        m_b = b;
        m_c = c;
        m_h = h;
        m_w = w;
    }

    // Utility to print the tensor for debugging
    void print(int items_to_print = 10) const {
        size_t num_elements = this->size();
        if (num_elements == 0 || m_ptr == nullptr) {
            std::cout << "Tensor is empty." << std::endl;
            return;
        }

        // Ensure we don't try to print more elements than exist.
        items_to_print = std::min((int)num_elements, items_to_print);

        std::vector<float> host_data(items_to_print);
        CUDA_CHECK(cudaMemcpy(host_data.data(), m_ptr, items_to_print * sizeof(float), cudaMemcpyDeviceToHost));

        std::cout << "Tensor (Shape: " << m_b << "x" << m_c << "x" << m_h << "x" << m_w << ") - "
                  << "First " << items_to_print << " of " << num_elements << " elements:" << std::endl;
        
        for (int i = 0; i < items_to_print; ++i) {
            std::cout << host_data[i] << " ";
        }
        std::cout << std::endl;
    }


private:
    float* m_ptr;
    bool m_owns_memory; // Flag to control whether the destructor calls cudaFree
    int m_b, m_c, m_h, m_w;
};