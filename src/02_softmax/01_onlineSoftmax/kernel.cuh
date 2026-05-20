#pragma once
#include <algorithm>
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define FULL_MASK 0xffffffffu // unsigned, safer in bit shifting

namespace FlashLab::Softmax {

    void launch_softmax_v1 (float *input, float *output, const int d, cudaStream_t stream);

    template <typename T, const int BN>
    void launch_softmax_v2 (void *input, void *output, const int N, const int d, cudaStream_t stream) {
        dim3 gridDim(CEIL_DIV(N, BN));
        dim3 blockDim(BN * 32);
        
        switch(d) {
            case 64:
                softmax_v2<T, 64, BN><<<gridDim, blockDim, 0, stream>>>(input, output, N, d); break;
            case 128:
                softmax_v2<T, 128, BN><<<gridDim, blockDim, 0, stream>>>(input, output, N, d); break;
            case 256:
                softmax_v2<T, 256, BN><<<gridDim, blockDim, 0, stream>>>(input, output, N, d); break;
            case 512:
                softmax_v2<T, 512, BN><<<gridDim, blockDim, 0, stream>>>(input, output, N, d); break;
            case 1024:
                softmax_v2<T, 1024, BN><<<gridDim, blockDim, 0, stream>>>(input, output, N, d); break;
            default:
                std::printf("Unsupported dimension d=%d. Add it to the switch!", d);
        }

        // Check for launch errors (like passing a CPU pointer!)
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            printf("Kernel Launch Error: %s\n", cudaGetErrorString(err));
    }

}
