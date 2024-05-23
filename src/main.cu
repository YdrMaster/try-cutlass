#include "cast.cuh"
#include "compare.cuh"
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

    float cublas_time = 0, our_time = 0;

    Cublas cublas(stream);
    for (size_t i = 0; i < repeat; i++) {
        float ms;

        CUDA_CALL(cudaEventRecord(start, stream));
        cublas.gemm(c_cublas, a, b, m, n, k);
        CUDA_CALL(cudaEventRecord(end, stream));

        CUDA_CALL(cudaEventSynchronize(end));
        CUDA_CALL(cudaEventElapsedTime(&ms, start, end));
        cublas_time += ms;

        CUDA_CALL(cudaEventRecord(start, stream));
        gemm_simple(c_our, a, b, m, n, k, stream);
        CUDA_CALL(cudaEventRecord(end, stream));

        CUDA_CALL(cudaEventSynchronize(end));
        CUDA_CALL(cudaEventElapsedTime(&ms, start, end));
        our_time += ms;
    }
    printf("cublas time: %.3f ms\n", cublas_time / repeat);
    printf("our time: %.3f ms\n", our_time / repeat);

    compare(c_cublas, c_our, 1e-3, m * n);

    {
        std::vector<half> host(m * k);
        CUDA_CALL(cudaMemcpy(host.data(), a, m * k * sizeof(half), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaFree(a));
        printf("A %ldx%ld\n", m, k);
        // for (size_t i = 0; i < m; i++) {
        //     for (size_t j = 0; j < k; j++) {
        //         printf("%.3f ", (float) host[i * k + j]);
        //     }
        //     printf("\n");
        // }
    }
    {
        std::vector<half> host(k * n);
        CUDA_CALL(cudaMemcpy(host.data(), b, k * n * sizeof(half), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaFree(b));
        printf("B %ldx%ld\n", k, n);
        // for (size_t i = 0; i < k; i++) {
        //     for (size_t j = 0; j < n; j++) {
        //         printf("%.3f ", (float) host[i + j * k]);
        //     }
        //     printf("\n");
        // }
    }
    {
        std::vector<half> host(m * n);
        CUDA_CALL(cudaMemcpy(host.data(), c_cublas, m * n * sizeof(half), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaFree(c_cublas));
        printf("C CUBLAS %ldx%ld\n", m, n);
        // for (size_t i = 0; i < m; i++) {
        //     for (size_t j = 0; j < n; j++) {
        //         printf("%.3f ", (float) host[i * n + j]);
        //     }
        //     printf("\n");
        // }
    }
    {
        std::vector<half> host(m * n);
        CUDA_CALL(cudaMemcpy(host.data(), c_our, m * n * sizeof(half), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaFree(c_our));
        printf("C CUTLASS %ldx%ld\n", m, n);
        // for (size_t i = 0; i < m; i++) {
        //     for (size_t j = 0; j < n; j++) {
        //         printf("%.3f ", (float) host[i * n + j]);
        //     }
        //     printf("\n");
        // }
    }

    CUDA_CALL(cudaEventDestroy(start));
    CUDA_CALL(cudaEventDestroy(end));
    CUDA_CALL(cudaStreamDestroy(stream));
}

int main(int argc, char **argv) {
    if (argc != 4) {
        printf("Usage: %s m n k\n", argv[0]);
        return EXIT_FAILURE;
    }

    size_t m = atoi(argv[1]);
    size_t n = atoi(argv[2]);
    size_t k = atoi(argv[3]);
    test(m, n, k, 100);
    return EXIT_SUCCESS;
}
