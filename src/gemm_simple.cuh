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

    clear(tCrC);

    auto num_tile_k = size<2>(a_local);
#pragma unroll 1
    for (int itile = 0; itile < num_tile_k; ++itile) {
        cute::copy(tAgA(_, _, _, itile), tArA);
        cute::copy(tBgB(_, _, _, itile), tBrB);

        cute::gemm(tiled_mma, tCrC, tArA, tBrB, tCrC);
    }

    cute::copy(tCrC, tCgC);
}

template<class T, size_t TM_K, size_t TN_K, size_t TK_K>
void gemm_simple(
    half *c,
    half const *a,
    half const *b,
    size_t m, size_t n, size_t k,
    cudaStream_t stream) {

    using mma_op = SM80_16x8x16_F16F16F16F16_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;

    using MMA = decltype(make_tiled_mma(
        mma_atom{},
        make_layout(Shape<_2, _2, _1>{}),
        make_layout(Shape<_1, _2, _1>{})));
    dim3 block(size(MMA{}));
    dim3 grid(n / TN_K, m / TM_K);

    gemm_simple_kernel<T, TM_K, TN_K, TK_K, MMA>
        <<<grid, block, 0, stream>>>(c, a, b, m, n, k);
}
