#ifndef __COMPARE_CUH__
#define __COMPARE_CUH__

#include "common.cuh"

struct CompareResult {
    int count;
    float max_error;
};

template<class T>
static __global__ void gpu_compare_kernel(
    T const *__restrict__ x,
    T const *__restrict__ y,
    CompareResult *result,
    float threshold,
    size_t n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n) {
        return;
    }

    float v0 = x[idx];
    float v1 = y[idx];

    float diff = fabs(v0 - v1);
    if (diff > threshold) {
        atomicAdd(&result->count, 1);

        // for positive floating point, there int representation is in the same order.
        int int_diff = *((int *) (&diff));
        atomicMax((int *) &result->max_error, int_diff);
    }
}

template<class T>
void compare(T const *x, T const *y, float threshold, int n) {
    CompareResult host, *dev;
    CUDA_CALL(cudaMalloc(&dev, sizeof(dev)));

    constexpr size_t THREADS_PER_BLOCK = 1024;

    auto grid = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    auto block = (n + grid - 1) / grid;

    gpu_compare_kernel<<<grid, block>>>(x, y, dev, threshold, n);

    CUDA_CALL(cudaMemcpy(&host, dev, sizeof(host), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaFree(dev));

    if (host.count == 0) {
        printf("check ok, max_error = %f\n",
               host.max_error);
    } else {
        printf("==============================="
               "check fail: diff %.1f%% = %d/%d max_error = %f"
               "===============================\n",
               (100.f * host.count) / n,
               host.count, n,
               host.max_error);
    }
}

#endif// __COMPARE_CUH__
