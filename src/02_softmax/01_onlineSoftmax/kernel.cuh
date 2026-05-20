#pragma once
#include <algorithm>
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string> // for std::string

#define FULL_MASK 0xffffffffu // unsigned, safer in bit shifting

namespace FlashLab::Softmax {

    void launch_softmax_v1 (float *input, float *output, const int d, cudaStream_t stream);

    void launch_softmax_v2 (float *input, float *output, const int N, const int d, cudaStream_t stream);

}
