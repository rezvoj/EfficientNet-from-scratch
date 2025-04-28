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
