#pragma once
#include <stdexcept>
#include <cudnn_v9.h>
#include <cublas_v2.h> 



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


class CudnnException : public std::runtime_error {
public:
    CudnnException(const cudnnStatus_t status) :
        std::runtime_error(cudnnGetErrorString(status)) {}
    CudnnException(const std::string& message) :
        std::runtime_error(message) {}
};


__forceinline__
void checkCudnn(const cudnnStatus_t status) {
    if (status != CUDNN_STATUS_SUCCESS) {
        throw CudnnException(status);
    }
}


class CublasException : public std::runtime_error {
public:
    CublasException(const cublasStatus_t status) :
        std::runtime_error(cublasGetStatusString(status)) {}
};


__forceinline__
void checkCublas(const cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw CublasException(status);
    }
}
