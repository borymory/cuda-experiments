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
        // For simplicity assume Br = Bc = 32. Then Ill check if it works for Br = 32, Bc = 64.
        int rowIdx = blockIdx.x * Br;

        // advance each block to q and o blocks
        Q += rowIdx * d;
        O += rowIdx * d;

        int tx = threadIdx.x % 32;
        int ty = threadIdx.x / 32;

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
            Q_i[ty * D + (tx + offset)] = Q[ty * D + (tx + offset)];
        }
        __syncthreads();

        // start for loop for K_j V_j loading
        for (unsigned int outerLoop = 0; outerLoop < N; outerLoop += Bc) {

            for (unsigned int offset = 0; offset < d; offset += 32) {
                // assumption: d%32=0
                K_j[ty * D + (tx + offset)] = K[ty * D + (tx + offset)];
                V_j[ty * D + (tx + offset)] = V[ty * D + (tx + offset)];
            }
            __syncthreads();

            // calculate S=Q@K^T.
            for(unsigned int offset = 0; offset < Bc; offset += 32) {
                // assumption: Bc, Br%32=0
                float qk_sum = 0.0f;
                for (unsigned int k = 0; k < d; ++k) {
                    float q_value = Q_i[ty * D + k];
                    float k_value = K_j[(tx + offset) * D + k];     // possible bank conflict at K_j accesses
                    qk_sum += q_value * k_value;
                }
                S_ij[ty * Bc + (tx + offset)] = qk_sum;
            }
            __syncthreads();    // make sure all threads finish S_ij load

            // thread-level softmax: reduce S_ij into register for online softmax
            float d_i = 0.0f;
            float m_i = -INFINITY;
            for (unsigned int offset = 0; offset < Bc; offset += 32) {
                // assumption: Bc, Br%32=0
                int dataIdx = ty * Bc + (tx + offset);
                float val = S_ij[dataIdx];

                float m_prev = m_i;
                m_i = fmaxf(m_i, val);      // obtain new max
                d_i *= expf(m_prev - m_i);  // scale old norm
                d_i += expf(val - m_i);     // add running contribution
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
                // assumption: Bc, Br%32=0
                int dataIdx = ty * Bc + (tx + offset);
                float s_val = S_ij[dataIdx];
                S_ij[dataIdx] = expf(s_val - m_i);
            }
            __syncthreads();   // wait for load before PV matmul

            // now, m_i and d_i are block max and norms
            
            // calculate new global max and norms: (current stats + prev softmax result)
            float m_new = fmaxf(m_i, m_old);
            float d_new = d_old * expf(m_old - m_new) + d_i * expf(m_i - m_new);
            
            // matmul PV, load into O
            // Br by 32 chunks of O_i are calculated
            #pragma unroll
            for (unsigned int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                // assumption: Bc, Br%32=0
                float pv_sum = 0.0f;
                int p_dataIdx = ty * Bc;          // same rowIdx, size: (Br, Bc)
                int v_dataIdx = (tx * 32 * i);    // v is tiled by stride 32, size (Bc, d)
                for (unsigned int k = 0; k < Bc; ++k) {
                    float p_val = S_ij[p_dataIdx + k];
                    float v_val = V_j[k * D + v_dataIdx];
                    pv_sum += p_val * v_val;
                }
                O_reg[i] *= d_old * (1/d_new) * expf(m_old - m_new);    // rescale old block to new global stats
                O_reg[i] += (1/d_new) * pv_sum * expf(m_i - m_new);     // add current block (scaled now)
            }

            // save current globals as prev blocks
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
            int o_dataIdx = ty * D + (tx + 32 * i);
            O[o_dataIdx] = O_reg[i];
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
            case(32):
                flashAttn_fwd_v1<Br, Bc, 32><<<gridDim, blockDim, 0, stream>>>(K, Q, V, O, N, d); break;
        }

        // Check for launch errors (like passing a CPU pointer!)
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            printf("Kernel Launch Error: %s\n", cudaGetErrorString(err));
    }

}