#include "kernel.cuh"
#include "utils.cuh"

namespace FlashLab::Softmax {
    //
    // OVERVIEW: 
    //  * softmax_v1 calculates online softmax for a signle row vector of length d. Assumption: d % 32 = 0.
    //  * softmax_v2 calculates online softmax for a matrix of size N * d. Assumption: d % 32 = 0, N % BN = 0.

    //
    // KERNELS
    //
    template <const int D>
    __global__ void softmax_v1 (float *input, float *output, const int d) {
        // Assume d >> 32
        int tx = threadIdx.x;
        const int ELEMENTS_PER_THREAD = D/32;

        // Register level statistics and variables
        float threadRes[ELEMENTS_PER_THREAD];   // used only for storing input values
        float d_i = 0.0f;
        float m_i = -INFINITY;

        // Thread-Level softmax: Reduce vector into registers while doing online softmax
        #pragma unroll
        for (unsigned int resIdx = 0; resIdx < ELEMENTS_PER_THREAD; ++resIdx) {
            // Assuming d%32=0
            int dataIdx = tx + resIdx * 32;
            float val = input[dataIdx];

            threadRes[resIdx] = val;        // store in register array

            float m_prev = m_i;
            m_i = fmaxf(m_prev, val);       // obtain new max
            d_i *= expf(m_prev - m_i);      // scale sum by new max
            d_i += expf(val - m_i);         // add running contribution
        }

        // Warp-Level softmax: Use __shfl_xor_sync for each thread to exchange statistics and update themselves
        for (unsigned int mirrorIdx = 1; mirrorIdx <= 16; mirrorIdx <<= 1) {
            float m_j = __shfl_xor_sync(FULL_MASK, m_i, mirrorIdx);     // obtain m_j from another thread
            float d_j = __shfl_xor_sync(FULL_MASK, d_i, mirrorIdx);     // obtain d_j from another thread

            float max = fmaxf(m_i, m_j);    // max = max(m_i, m_j)
            d_i *= expf(m_i - max);         // rescale old sum
            d_i += d_j * expf(m_j - max);   // add contribution from the new sum
            m_i = max;                      // update new max
        }

        // Calculate final values and write back
        #pragma unroll
        for (unsigned int resIdx = 0; resIdx < ELEMENTS_PER_THREAD; ++resIdx) {
            output[tx + (resIdx * 32)] = expf(threadRes[resIdx] - m_i) / d_i;
        }
    }

    template <const int D, const int BN>
    __global__ void softmax_v2 (float *input, float *output, const int N, const int d) {
        // Assume d >> 32
        // We launch CEIL_DIV(N, BN) many blocks

        // Position blocks to rows
        int rowIdx = blockIdx.x * BN;
        input += rowIdx * d;

        int tx = threadIdx.x % 32;
        int ty = threadIdx.x / 32;
        // Assuming N%32 = 0
        const int ELEMENTS_PER_THREAD = D/32;

        // Register level statistics and variables
        float threadRes[ELEMENTS_PER_THREAD];   // used only for storing input values
        float d_i = 0.0f;
        float m_i = -INFINITY;

        // Thread-Level softmax: Reduce vector into registers while doing online softmax
        #pragma unroll
        for (unsigned int resIdx = 0; resIdx < ELEMENTS_PER_THREAD; ++resIdx) {
            // Assuming d%32=0
            int dataIdx = ty * d + (tx + resIdx * 32);
            float val = input[dataIdx];

            threadRes[resIdx] = val;        // store in register array

            float m_prev = m_i;
            m_i = fmaxf(m_prev, val);       // obtain new max
            d_i *= expf(m_prev - m_i);      // scale sum by new max
            d_i += expf(val - m_i);         // add running contribution
        }

        // Warp-Level softmax: Use __shfl_xor_sync for each thread to exchange statistics and update themselves
        for (unsigned int mirrorIdx = 1; mirrorIdx <= 16; mirrorIdx <<= 1) {
            float m_j = __shfl_xor_sync(FULL_MASK, m_i, mirrorIdx);     // obtain m_j from another thread
            float d_j = __shfl_xor_sync(FULL_MASK, d_i, mirrorIdx);     // obtain d_j from another thread

            float max = fmaxf(m_i, m_j);    // max = max(m_i, m_j)
            d_i *= expf(m_i - max);         // rescale old sum
            d_i += d_j * expf(m_j - max);   // add contribution from the new sum
            m_i = max;                      // update new max
        }

        // Calculate final values and write back
        #pragma unroll
        for (unsigned int resIdx = 0; resIdx < ELEMENTS_PER_THREAD; ++resIdx) {
            output[ty * d + (tx + resIdx * 32)] = expf(threadRes[resIdx] - m_i) / d_i;
        }
    }

    //
    // KERNEL WRAPPERS
    //
    void launch_softmax_v1 (float *input, float *output, const int d, cudaStream_t stream) {
        dim3 gridDim(1);
        dim3 blockDim(32);
        
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

    void launch_softmax_v2 (float *input, float *output, const int N, const int d, cudaStream_t stream) {
        const int BN = 8;
        dim3 gridDim(CEIL_DIV(N, BN));
        dim3 blockDim(BN * 32);
        
        switch(d) {
            case 128:
                softmax_v2<128, BN><<<gridDim, blockDim, 0, stream>>>(input, output, N, d);
                break;
            case 256:
                softmax_v2<256, BN><<<gridDim, blockDim, 0, stream>>>(input, output, N, d);
                break;
            case 512:
                softmax_v2<512, BN><<<gridDim, blockDim, 0, stream>>>(input, output, N, d);
                break;
            case 1024:
                softmax_v2<1024, BN><<<gridDim, blockDim, 0, stream>>>(input, output, N, d);
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

// TODO: The switch trick above must be explained in softmax.md, or another .md file such as performance.md. Also include:
//       It is important to disscuss templates effect on compile time and runtime. What is the difference between
//       local arrays stored adress when accessed with pragma unroll + template or by index? Local mem vs Registers.
//       Why switch case is implemented to adress the problem above? "the perfect blocksize prevents spilling!"
//       "If the compiler can prove that the index is a constant at compile-time, it will put the array in Registers."