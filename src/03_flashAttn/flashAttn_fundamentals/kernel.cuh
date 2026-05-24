#pragma once
#include <algorithm>
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string> // for std::string

#define FULL_MASK 0xffffffffu // unsigned, safer in bit shifting

namespace FlashLab::flashAttn::naive {

    void launch_flashAttn_fwd_v1(float *K, float *Q, float *V, float *O, const int N, const int d, cudaStream_t stream);

}
