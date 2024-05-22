#ifndef __COMMON_CUH__
#define __COMMON_CUH__

#include <cstdio>

#define PANIC(EXPR)                                             \
    printf("Error at %s:%d - %s\n", __FILE__, __LINE__, #EXPR); \
    exit(EXIT_FAILURE)

#define CUDA_CALL(x)              \
    do {                          \
        if ((x) != cudaSuccess) { \
            PANIC(x);             \
        }                         \
    } while (0)


#endif// __COMMON_CUH__
