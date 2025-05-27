#pragma once
#include <stdexcept>



class CudaException : public std::runtime_error {
    public: CudaException(const cudaError_t error): 
        std::runtime_error(cudaGetErrorString(error)) {}
};


__forceinline__
void checkCuda(const cudaError_t err) {
    if (err != cudaSuccess) {
        throw CudaException(err);
    }
}


__forceinline__
void checkCudaLastError() {
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw CudaException(err);
    }
}
