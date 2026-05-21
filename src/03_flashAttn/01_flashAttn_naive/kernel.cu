#include "kernel.cuh"
#include "utils.cuh"

namespace FlashLab::flashAttn::naive {
    //
    // OVERVIEW: 
    //  * flashAttn_fwd_v1: no mask, no dropout. S_ij small tile stored in shared memory. Later on will try to do it without smem, but registers.

    //
    // KERNELS
    //
    template<const int Br, const int Bc>
    __global__ void flashAttn_fwd_v1 (float *K, float *Q, float *V, float *O, const int N, const int d) {
        // We launch Br * 32 threads. SMEM loading is tiled
        int rowIdx = blockIdx.x * Br;
        Q += rowIdx * d;    // advance each block to q and o blocks
        O += rowIdx * d;

        int tx_i = threadIdx.x % 32;
        int ty_i = threadIdx.x / 32;

        int tx_j = (threadIdx.x * Bc) % (Br * 32);
        int ty_j = (threadIdx.x * Bc) / (Br * 32);

        __shared__ float Q_i [Br * d];
        __shared__ float K_j [Bc * d];
        __shared__ float V_j [Bc * d];
        __shared__ float S_ij [Br * Bc];

        for (unsigned int offset = 0; offset < d; offset += 32) {
            // assumption: d%32=0
            Q_i[ty_i * d + (tx_i + offset)] = Q[ty_i * d + (tx_i + offset)];
        }

        // start for loop for K_j V_j loading
        for (unsigned int offset = 0; offset < d; offset += (Br * 32) / Bc) {
            // assumption: d%( (Br * 32) / Bc )=0
            K_j[ty_j * d + (tx_j + offset)] = K[ty_j * d + (tx_j + offset)];
            V_j[ty_j * d + (tx_j + offset)] = V[ty_j * d + (tx_j + offset)];
        }

        // calculate S=Q@K^T
        for(unsigned int offset = 0; offset < d; offset += 32) {
            float sum = 0.0f;
            for (unsigned int dotIdx = 0; dotIdx < d; ++dotIdx) {
                sum += Q_i[ty_i * d + dotIdx] * K_j[(ty_j + offset) * d + dotIdx];
            }
            S_ij[ty_i * d + (ty_j + offset)] = sum;
        }

        // thread-level softmax: reduce S_ij into register for online softmax
        float d_i = 0.0f;
        float m_i = -INFINITY;
        for (unsigned int offset = 0; offset < Bc; offset += 32) {
            // assumption: Bc%32 = 0, Bc >> 32
            int dataIdx = ty_i * Bc + (tx_i + offset);
            float val = S_ij[dataIdx];

            float m_prev = m_i;
            m_i = fmaxf(m_i, val);
            d_i *= expf(m_prev - m_i);
            d_i += expf(val - m_i);
        }

        // warp-level softmax
        for (unsigned int mirrorIdx = 1; mirrorIdx <= 16; mirrorIdx <<= 1) {
            float m_j = __shfl_xor_sync(FULL_MASK, m_i, mirrorIdx); // obtain m_j
            float d_j = __shfl_xor_sync(FULL_MASK, d_i, mirrorIdx); // obtain d_j

            float max = fmaxf(m_i, m_j);    // find new max
            d_i *= expf(m_i - max);         // rescale old sum
            d_i += d_j * expf(m_j - max);   // add contribution from current sum
            m_i = max;
        }

        // calculate P_ij
        for (unsigned int offset = 0; offset < Bc; offset += 32) {
            // assumption: Bc%32 = 0, Bc >> 32
            int dataIdx = ty_i * Bc + (tx_i + offset);
            float val = S_ij[dataIdx];
            S_ij[dataIdx] = expf(val - m_i) / d_i;
        }

        // Matmul PV, load into O + update old stats

        // finish for loop



    }
    

    //
    // KERNEL WRAPPERS
    //
    

}