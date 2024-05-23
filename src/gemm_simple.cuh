#ifndef __GEMM_SIMPLE_CUH__
#define __GEMM_SIMPLE_CUH__

#include <cute/tensor.hpp>

using namespace cute;

template<typename T, size_t TM_K, size_t TN_K, size_t TK_K, class TiledMMA>
static __global__ void gemm_simple_kernel(
    T *__restrict__ c_,
    T const *__restrict__ a_,
    T const *__restrict__ b_,
    size_t m, size_t n, size_t k) {

    Tensor c = make_tensor(make_gmem_ptr(c_), make_shape(m, n), make_stride(n, Int<1>{}));
    Tensor a = make_tensor(make_gmem_ptr(a_), make_shape(m, k), make_stride(k, Int<1>{}));
    Tensor b = make_tensor(make_gmem_ptr(b_), make_shape(n, k), make_stride(k, Int<1>{}));

    auto ix = blockIdx.x,
         iy = blockIdx.y;

    Tensor c_local = local_tile(c, make_tile(Int<TM_K>{}, Int<TN_K>{}), make_coord(iy, ix));
    Tensor a_local = local_tile(a, make_tile(Int<TM_K>{}, Int<TK_K>{}), make_coord(iy, _));
    Tensor b_local = local_tile(b, make_tile(Int<TN_K>{}, Int<TK_K>{}), make_coord(ix, _));

    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    auto tCgC = thr_mma.partition_C(c_local);// (MMA, MMA_M, MMA_N)
    auto tAgA = thr_mma.partition_A(a_local);// (MMA, MMA_M, MMA_K, num_tile_k)
    auto tBgB = thr_mma.partition_B(b_local);// (MMA, MMA_N, MMA_K, num_tile_k)

    auto tCrC = thr_mma.partition_fragment_C(c_local(_, _));   // (MMA, MMA_M, MMA_N)
    auto tArA = thr_mma.partition_fragment_A(a_local(_, _, 0));// (MMA, MMA_M, MMA_K)
    auto tBrB = thr_mma.partition_fragment_B(b_local(_, _, 0));// (MMA, MMA_N, MMA_K)

    // auto tCrC = make_tensor<float>(tCrC_.layout());
    clear(tCrC);
    int num_tile_k = size<2>(a_local);
    for (int itile = 0; itile < num_tile_k; ++itile) {
        cute::copy(tAgA(_, _, _, itile), tArA);
        cute::copy(tBgB(_, _, _, itile), tBrB);
        cute::gemm(tiled_mma, tArA, tBrB, tCrC);
    }

    cute::copy(tCrC, tCgC);
}

void gemm_simple(
    half *c,
    half const *a,
    half const *b,
    size_t m, size_t n, size_t k,
    cudaStream_t stream) {
    constexpr static size_t
        TM_K = 128,
        TN_K = 128,
        TK_K = 32;

    auto mma = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{});
    dim3 grid(n / TN_K, m / TM_K);
    dim3 block(size(mma));

    printf("grid: %d, %d, block: %d\n", grid.x, grid.y, block.x);

    gemm_simple_kernel<half, TM_K, TN_K, TK_K, decltype(mma)>
        <<<grid, block, 0, stream>>>(c, a, b, m, n, k);
}

#endif// __GEMM_SIMPLE_CUH__
