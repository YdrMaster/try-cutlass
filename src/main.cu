#include "cast.cuh"
#include "cublas.cuh"
#include "gemm_simple.cuh"
#include "rng.cuh"

void test(size_t m, size_t n, size_t k, size_t repeat) {
    half *c_cublas, *c_our, *a, *b;
    CUDA_CALL(cudaMalloc(&c_cublas, m * n * sizeof(half)));
    CUDA_CALL(cudaMalloc(&c_our, m * n * sizeof(half)));
    CUDA_CALL(cudaMalloc(&a, m * k * sizeof(half)));
    CUDA_CALL(cudaMalloc(&b, k * n * sizeof(half)));

    RNG gen(1234ULL);
    float *rand;
    CUDA_CALL(cudaMalloc(&rand, std::max(m * k, k * n) * sizeof(float)));
    gen.rand(rand, m * k);
    cast(a, rand, m * k);
    gen.rand(rand, k * n);
    cast(b, rand, k * n);

    cudaStream_t stream;
    CUDA_CALL(cudaStreamCreate(&stream));

    cudaEvent_t start, end;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&end));

    float cublas_time = 0.0;
    Cublas cublas(stream);
    for (size_t i = 0; i < repeat; i++) {
        CUDA_CALL(cudaEventRecord(start, stream));
        cublas.gemm(c_cublas, a, b, m, n, k);
        CUDA_CALL(cudaEventRecord(end, stream));

        CUDA_CALL(cudaEventSynchronize(end));
        float ms;
        CUDA_CALL(cudaEventElapsedTime(&ms, start, end));
        cublas_time += ms;
    }
    printf("cublas time: %.3f ms\n", cublas_time / repeat);

    float our_time = 0.0;
    for (size_t i = 0; i < repeat; i++) {
        CUDA_CALL(cudaEventRecord(start, stream));
        gemm_simple<half, 128, 128, 32>(c_our, a, b, m, n, k, stream);
        CUDA_CALL(cudaEventRecord(end, stream));

        CUDA_CALL(cudaEventSynchronize(end));
        float ms;
        CUDA_CALL(cudaEventElapsedTime(&ms, start, end));
        our_time += ms;
    }
    printf("our time: %.3f ms\n", our_time / repeat);

    CUDA_CALL(cudaEventDestroy(start));
    CUDA_CALL(cudaEventDestroy(end));
    CUDA_CALL(cudaStreamDestroy(stream));
}

int main(int argc, char **argv) {
    test(81920, 256, 256, 100);
    return EXIT_SUCCESS;
}
