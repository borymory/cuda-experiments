#pragma once
#include <algorithm>
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string> // for std::string

#define FULL_MASK 0xffffffffu // unsigned, safer in bit shifting

namespace FlashLab::flashAttn::fundamentals {

    void launch_QK_matmul (float *K, float *Q, float *S, const int N, const int d, cudaStream_t stream);

    void launch_S_softmax (float *K, float *Q, float *S, const int N, const int d, cudaStream_t stream);

}
