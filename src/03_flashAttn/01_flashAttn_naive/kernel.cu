#include "kernel.cuh"
#include "utils.cuh"

namespace FlashLab::flashAttn::naive {
    //
    // OVERVIEW: 
    //  * flashAttn_fwd_v1: no mask, no dropout. S_ij small tile stored in shared memory. Later on will try to do it without smem, but registers.

    //
    // KERNELS
    //
    template<const int Br, const int Bc, const int D>
    __global__ void flashAttn_fwd_v1 (float *K, float *Q, float *V, float *O, const int N, const int d) {
        // We launch Br * 32 threads. SMEM loading is tiled
        // We launch CEIL_DIV(N, Br) many blocks
        int rowIdx = blockIdx.x * Br;

        // advance each block to q and o blocks
        Q += rowIdx * d;
        O += rowIdx * d;

        int tx_i = threadIdx.x % 32;
        int ty_i = threadIdx.x / 32;

        int tx_j = (threadIdx.x) % (32 * Br / Bc);
        int ty_j = (threadIdx.x) / (32 * Br / Bc);

        __shared__ float Q_i [Br * D];
        __shared__ float K_j [Bc * D];
        __shared__ float V_j [Bc * D];
        __shared__ float S_ij [Br * Bc];

        // threads' O elements responsibility
        const int ELEMENTS_PER_THREAD = D/32;
        float O_reg[ELEMENTS_PER_THREAD] = {0.0f};

        // prev block statistics
        float m_old = -INFINITY;
        float d_old = 0.0f;

        for (unsigned int offset = 0; offset < d; offset += 32) {
            // assumption: d%32=0
            Q_i[ty_i * D + (tx_i + offset)] = Q[ty_i * D + (tx_i + offset)];
        }
        __syncthreads();

        // start for loop for K_j V_j loading
        for (unsigned int outerLoop = 0; outerLoop < N; outerLoop += Bc) {

            for (unsigned int offset = 0; offset < d; offset += (Br * 32) / Bc) {
                // assumption: d%( (Br * 32) / Bc )=0
                K_j[ty_j * D + (tx_j + offset)] = K[ty_j * D + (tx_j + offset)];
                V_j[ty_j * D + (tx_j + offset)] = V[ty_j * D + (tx_j + offset)];
            }
            __syncthreads();

            // calculate S=Q@K^T.
            for(unsigned int offset = 0; offset < Bc; offset += 32) {
                // assumption: Bc%32 = 0, Bc >> 32
                float sum = 0.0f;
                for (unsigned int dotIdx = 0; dotIdx < d; ++dotIdx)
                    sum += Q_i[ty_i * D + dotIdx] * K_j[(tx_i + offset) * D + dotIdx];  // possible bank conflict at K_j accesses
                S_ij[ty_i * Bc + (tx_i + offset)] = sum;
            }
            __syncthreads();

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
                S_ij[dataIdx] = expf(val - m_i);
            }
            __syncthreads();

            // now, m_i and d_i are block max and norms

            // global max and norms: calculate new stats (current softmax + prev softmax results)
            float m_new = fmaxf(m_i, m_old);
            float d_new = d_old * expf(m_old - m_new) + d_i * expf(m_i - m_new);
            
            // matmul PV, load into O
            #pragma unroll
            for (unsigned int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                // assumption: Bc%32=0, Bc >> 32
                float pv_sum = 0.0f;
                for (unsigned int dotIdx = 0; dotIdx < Bc; ++dotIdx) {
                    float p_val = S_ij[ty_i * Bc + dotIdx];
                    float v_val = V_j[dotIdx * D (tx_i + (32 * i))];
                    pv_sum += p_val * v_val;
                }
                O_reg[i] *= d_old * (1/d_new) * expf(m_old - m_new);
                O_reg[i] += (1/d_new) * pv_sum * expf(m_i - m_new);
            }

            // save new globals as prev blocks
            m_old = m_new;
            d_old = d_new;

            // advance to the next block
            K += Bc * d;
            V += Bc * d;

            // finish for loop
        }

        // write results
        #pragma unroll
        for (unsigned int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
            // assumption: d%32=0, d >> 32
            O[ty_i * D + (tx_i + 32 * i)] = O_reg[i];
        }

    }
    

    //
    // KERNEL WRAPPERS
    //
    void launch_flashAttn_fwd_v1(float *K, float *Q, float *V, float *O, const int N, const int d, cudaStream_t stream) {
        const int Br = 32;
        const int Bc = 32;
        dim3 gridDim(CEIL_DIV(N, Br));
        dim3 blockDim(32 * Br);
        
        switch(d) {
            case(64):
                flashAttn_fwd_v1<Br, Bc, 64><<<gridDim, blockDim, 0, stream>>>(K, Q, V, O, N, d); break;
            case(128):
                flashAttn_fwd_v1<Br, Bc, 128><<<gridDim, blockDim, 0, stream>>>(K, Q, V, O, N, d); break;
        }
    }

}