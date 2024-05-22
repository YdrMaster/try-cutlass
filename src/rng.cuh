#ifndef __RNG_CUH__
#define __RNG_CUH__

#include "common.cuh"
#include <curand.h>

#define CURAND_CALL(x)                      \
    do {                                    \
        if ((x) != CURAND_STATUS_SUCCESS) { \
            PANIC(x);                       \
        }                                   \
    } while (0)

class RNG {
    curandGenerator_t _gen;

public:
    explicit RNG(size_t seed) {
        CURAND_CALL(curandCreateGenerator(&_gen, CURAND_RNG_PSEUDO_DEFAULT));
        CURAND_CALL(curandSetPseudoRandomGeneratorSeed(_gen, seed));
    }
    ~RNG() noexcept {
        CURAND_CALL(curandDestroyGenerator(_gen));
    }

    RNG(RNG const &) = delete;
    RNG(RNG &&) noexcept = delete;
    RNG &operator=(RNG const &) = delete;
    RNG &operator=(RNG &&) noexcept = delete;

    void rand(float *mem, size_t n) const {
        CURAND_CALL(curandGenerateUniform(_gen, mem, n));
    }
};

#endif// __RNG_CUH__
