#pragma once
#include <algorithm>
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define FULL_MASK 0xffffffffu // unsigned, safer in bit shifting

namespace FlashLab::Reduction {

    void test_vectorReduction_v2 (float *A, const int d, cudaStream_t stream);

    void test_vectorReductionXOR_v2 (float *A, const int d, cudaStream_t stream);

    void test_rowSumXOR_v2 (float *B, const int N, const int d, cudaStream_t stream);

}
