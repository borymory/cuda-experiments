#include "kernel.cuh"
#include "utils.cuh"

namespace FlashLab::Softmax {
    //
    // OVERVIEW: 
    //  * softmax_v1 calculates online softmax for a signle row vector of length d. Assumption: d % 32 = 0.

    //
    // KERNELS
    //
    __global__ void softmax_v1 (float *input, float *output, const ind d) {
        // Assume d >> 32
        int tx = threadIdx.x;

        // Necessary statistics
        float threadRes[d/32] = {0.0f};
        float d_i = 0.0f;
        float m_i = -INFINITY;
        float max = -INFINITY;
        float threadVal = 0.0f;

        // Reduce vec into registers: online softmax within a thread
        for (unsigned int offset = 0; offset < d; offset += 32) {
            // Assuming d%32=0
            int resIdx = 0;
            threadVal = input[tx + offset];
            threadRes[resIdx] = threadVal;
            d_i *= expf(m_i);               // scale by prev max
            m_i = fmaxf(m_i, threadVal);    // obtain new max
            d_i *= expf(-m_i);              // scale by new max
            d_i += expf(threadVal - m_i);   // add running contribution

            resIdx++
        }

        // online softmax between threads
        float m_j = -INFINITY;
        float d_j = 0.0f;
        for (unsigned int mirrorIdx = 1; mirrorIdx <= 16; mirrorIdx <<= 1) {
            m_j = __shfl_xor_sync(FULL_MASK, m_i, mirrorIdx);   // obtain m_j from another thread
            max = fmaxf(m_i, m_j);                              // max = max(m_i, m_j)
            d_j = __shfl_xor_sync(FULL_MASK, d_i, mirrorIdx);   // obtain d_j from another thread
            d_i *= expf(m_i - max);                             // rescale old sum
            d_i += d_j * expf(m_j - max);                       // add contribution from the new sum
        }

        // calc final values and write back
        for (unsigned int resIdx = 0; resIdx < d/32; ++resIdx) {
            threadRes[resIdx] = expf(threadRes[resIdx] - m_i) / d_i;
            output[resIdx * 32] = threadRes[resIdx];
        }
    }        

    //
    // KERNEL WRAPPERS
    //
    void launch_softmax_v1 (float *input, float *output, const int d, cudaStream_t stream) {
        dim3 gridDim = 1;
        dim3 blockDim = 32;

        softmax_v1<<<gridDim, blockDim, 0, stream>>>(input, output, d);
    }

}