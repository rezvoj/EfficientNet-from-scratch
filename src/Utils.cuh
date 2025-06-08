#pragma once
#include <stdexcept>
#include <cudnn.h>



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
};


__forceinline__
void checkCudnn(const cudnnStatus_t status) {
    if (status != CUDNN_STATUS_SUCCESS) {
        throw CudnnException(status);
    }
}


class NNException : public std::runtime_error {
    public: NNException(const std::string& errorMessage): 
        std::runtime_error(errorMessage) {}
};


__forceinline__
void cudaStartMeasuringTime(cudaEvent_t* start, cudaEvent_t* stop) {
    checkCuda(cudaEventCreate(start));
    checkCuda(cudaEventCreate(stop));
    checkCuda(cudaEventRecord(*start, 0));
}


__forceinline__
float cudaStopMeasuringTime(const cudaEvent_t start, const cudaEvent_t stop) {
    checkCuda(cudaEventRecord(stop, 0));
    checkCuda(cudaEventSynchronize(stop));
    float elapsedTime;
    checkCuda(cudaEventElapsedTime(&elapsedTime, start, stop));
    checkCuda(cudaEventDestroy(start));
    checkCuda(cudaEventDestroy(stop));
    return elapsedTime;
}


template <typename Integer>
__forceinline__ __host__ __device__
constexpr Integer ceilDiv(const Integer value, const Integer divisor) {
    return ((value + divisor - 1) / divisor);
}
