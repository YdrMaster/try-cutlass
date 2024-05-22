#ifndef __CAST_CUH__
#define __CAST_CUH__

#include "common.cuh"
#include <cute/tensor.hpp>

template<class T, class U, size_t ITEMS_PER_THREAD = 16 / std::max(sizeof(T), sizeof(U))>
static void __global__ cast_kernel(
    U *__restrict__ dst,
    T const *__restrict__ src,
    size_t n) {
    using namespace cute;

    auto tdst = make_tensor(make_gmem_ptr(dst), make_shape(n));
    auto tsrc = make_tensor(make_gmem_ptr(src), make_shape(n));

    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n / ITEMS_PER_THREAD) {
        return;
    }

    auto tdst_local = local_tile(tdst, make_shape(Int<ITEMS_PER_THREAD>{}), make_coord(idx));
    auto tsrc_local = local_tile(tsrc, make_shape(Int<ITEMS_PER_THREAD>{}), make_coord(idx));

    auto tdst_reg = make_tensor_like(tdst_local);
    auto tsrc_reg = make_tensor_like(tsrc_local);

    copy(tsrc_local, tsrc_reg);

#pragma unroll
    for (size_t i = 0; i < ITEMS_PER_THREAD; ++i) {
        tdst_reg(i) = (U) tsrc_reg(i);
    }

    copy(tdst_reg, tdst_local);
}

template<class T, class U>
void cast(U *dst, T const *src, size_t n) {
    constexpr size_t THREADS_PER_BLOCK = 1024;
    constexpr size_t ITEMS_PER_THREAD = 16 / std::max(sizeof(T), sizeof(U));

    if (n % ITEMS_PER_THREAD != 0) {
        PANIC(n must be divisible by ITEMS_PER_THREAD);
    }

    n /= ITEMS_PER_THREAD;
    auto block = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    auto threads = (n + block - 1) / block;
    n *= ITEMS_PER_THREAD;
    cast_kernel<<<block, threads>>>(dst, src, n);
}

#endif// __CAST_CUH__
