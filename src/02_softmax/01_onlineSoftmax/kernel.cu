#include "kernel.cuh"
#include "utils.cuh"

namespace FlashLab::Softmax {
    //
    // OVERVIEW: 
    //  * softmax_v1 calculates online softmax for a signle row vector of length d. Assumption: d % 32 = 0.

    //
    // KERNELS
    //
    template <const int D>
    __global__ void softmax_v1 (float *input, float *output, const int d) {
        // Assume d >> 32
        int tx = threadIdx.x;
        const int ELEMENTS_PER_THREAD = D/32;

        // Thread-level statistics
        float threadRes[ELEMENTS_PER_THREAD];
        float d_i = 0.0f;
        float m_i = -INFINITY;

        // Reduce vector into array and local online softmax
        #pragma unroll
        for (unsigned int resIdx = 0; resIdx < ELEMENTS_PER_THREAD; resIdx++) {
            // Assuming d%32=0
            int dataIdx = tx + resIdx * 32;
            float threadVal = input[dataIdx];

            threadRes[resIdx] = threadVal;      // store in register array

            float m_prev = m_i;
            m_i = fmaxf(m_prev, threadVal);     // obtain new max
            d_i *= expf(m_prev - m_i);          // scale sum by new max
            d_i += expf(threadVal - m_i);       // add running contribution
        }

        // Warp-Level reduction (Between threads)
        for (unsigned int mirrorIdx = 1; mirrorIdx <= 16; mirrorIdx <<= 1) {
            float m_j = __shfl_xor_sync(FULL_MASK, m_i, mirrorIdx);     // obtain m_j from another thread
            float d_j = __shfl_xor_sync(FULL_MASK, d_i, mirrorIdx);     // obtain d_j from another thread

            float max = fmaxf(m_i, m_j);                        // max = max(m_i, m_j)
            d_i *= expf(m_i - max);                             // rescale old sum
            d_i += d_j * expf(m_j - max);                       // add contribution from the new sum
        }

        // calc final values and write back
        #pragma unroll
        for (unsigned int resIdx = 0; resIdx < ELEMENTS_PER_THREAD; resIdx++) {
            output[tx + (resIdx * 32)] = expf(threadRes[resIdx] - m_i) / d_i;
        }
    }        

    //
    // KERNEL WRAPPERS
    //
    void launch_softmax_v1 (float *input, float *output, const int d, cudaStream_t stream) {
        dim3 gridDim = 1;
        dim3 blockDim = 32;
        
        switch(d) {
            case 128:
                softmax_v1<128><<<gridDim, blockDim, 0, stream>>>(input, output, d);
                break;
            case 256:
                softmax_v1<256><<<gridDim, blockDim, 0, stream>>>(input, output, d);
                break;
            case 512:
                softmax_v1<512><<<gridDim, blockDim, 0, stream>>>(input, output, d);
                break;
            case 1024:
                softmax_v1<1024><<<gridDim, blockDim, 0, stream>>>(input, output, d);
                break;
            default:
                std::printf("Unsupported dimension d=%d. Add it to the switch!", d);
        }

        // Check for launch errors (like passing a CPU pointer!)
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            printf("Kernel Launch Error: %s\n", cudaGetErrorString(err));
    }

}