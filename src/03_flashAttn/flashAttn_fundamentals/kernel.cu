#include "kernel.cuh"
#include "utils.cuh"

namespace FlashLab::flashAttn::fundamentals {
    //
    // Overview: Parts of the FlashAttn kernel that must be tested here before being fused in the main version
    //

    //
    // KERNELS
    //
    template<const int Br, const int Bc, const int D>
    __global__ void QK_matmul (float *K, float *Q, float *S, const int N, const int d) {
        // K is (N x d)
        // Q is (N x d)
        // out is (N x N)
        // We launch CEIL_DIV(N, Br) many blocks
        // We have Br * 32 threads per block
        // Each block takes a row-tile of Q, all loads row-blocks of K iteratively and does matmul QK^T,
        // similar to flashAttn.

        // For simplicity assume:
        // Bc >> Bc = 32
        // Bc % 32 = 0
        // N % Br = 0 (Less problems when assigning thread blocks to rows of S)
        // N % Bc = 0 (Less problems when iterating through chunks of K)
        // Bc % Br = 0 (Less problems when loading GMEM -> SMEM for K)

        int rowIdx = blockIdx.x * Br;   // General block offset (also for the O matrix)
        Q += rowIdx * D;                // advance pointer rn. => GMEM -> SMEM load indexology is easier to read
        S += rowIdx * N;                // advance S pointer rn. => SMEM -> GMEM loading would only require accounting for K_j block load offset


        int tx = threadIdx.x % 32;      // ranges 0 to 31
        int ty = threadIdx.x / 32;      // ranges 0 to Br-1

        __shared__ float Q_i[Br * D];
        __shared__ float K_j[Bc * D];
        __shared__ float S_ij[Br * Bc];

        // load Q_i
        for (unsigned int i = 0; i < D; i += 32) {
            
            int dataIdx = ty * D + (tx + i);
            Q_i[dataIdx] = Q[dataIdx];
        }
        __syncthreads();
        

        // add block load offset loop here
        for (unsigned int block_load_offset = 0; block_load_offset < N; block_load_offset += Bc) {
            
            // load K_j: a bit more complicated
            for (unsigned int k = 0; k < Bc; k += Br) {

                int outer_dIdx = ty + k;
                for (unsigned int i = 0; i < D; i += 32) {
                    int inner_dIdx = tx + i;
                    int dataIdx = outer_dIdx * D + inner_dIdx;
                    
                    K_j[dataIdx] = K[dataIdx];   
                }
            }
            __syncthreads();

            // matmul QK^T
            for (unsigned int i = 0; i < Bc; i += 32) {

                float qk_partial_sum = 0.0f;
                for (unsigned int k = 0; k < D; ++k) {
                    float q_val = Q_i[ty * D + k];
                    float k_val = K_j[(tx + i) * D + k];
                    qk_partial_sum += q_val * k_val;
                }
                S_ij[ty * Bc + (tx + i)] = qk_partial_sum;
            }
            __syncthreads();

            // load back to S
            for (unsigned int i = 0; i < Bc; i += 32) {
                int dIdx_GMEM = ty * N + (tx + i);
                int dIdx_SMEM = ty * Bc + (tx + i);
                S[dIdx_GMEM] = S_ij[dIdx_SMEM];
            }

            // advance blocks
            K += Bc * D;        // K is not transposed so block offset is along rows.
            S += Bc;            // The next block load will shift S along columns.
        }

    }

    template<const int Br, const int Bc, const int D>
    __global__ void S_softmax (float *K, float *Q, float *S, const int N, const int d) {
        // P = softmax(S)
        // Each warp does online softmax on a Br*Bc chunk of S_ij, writes back to S.
        // Online softmax is fused with the first matmul, directly doing as threads calculates
        // the S values (S_val). Then the regsiter reduced statistics are online softmaxed again by warp shuffle.
        // The results are Br, Bc chunk sizes of softmaxes written to GMEM.
        // To verify this kernel the CPU will do softmax Bc elements at a time, simulating the expected behaviour of the kernel :)

        int rowIdx = blockIdx.x * Br;   // General block offset (also for the O matrix)
        Q += rowIdx * D;                // advance pointer rn. => GMEM -> SMEM load indexology is easier to read
        S += rowIdx * N;                // advance S pointer rn. => SMEM -> GMEM loading would only require accounting for K_j block load offset


        int tx = threadIdx.x % 32;      // ranges 0 to 31
        int ty = threadIdx.x / 32;      // ranges 0 to Br-1

        __shared__ float Q_i[Br * D];
        __shared__ float K_j[Bc * D];
        __shared__ float S_ij[Br * Bc];

        // previous block statistics
        float m_old = -INFINITY;
        float d_old = 0.0f;

        // load Q_i
        for (unsigned int i = 0; i < D; i += 32) {
            
            int dataIdx = ty * D + (tx + i);
            Q_i[dataIdx] = Q[dataIdx];
        }
        __syncthreads();

        // add block load offset loop here
        for (unsigned int block_load_offset = 0; block_load_offset < N; block_load_offset += Bc) {
            
            // load K_j: a bit more complicated
            for (unsigned int k = 0; k < Bc; k += Br) {

                int outer_dIdx = ty + k;
                for (unsigned int i = 0; i < D; i += 32) {
                    int inner_dIdx = tx + i;
                    int dataIdx = outer_dIdx * D + inner_dIdx;
                    
                    K_j[dataIdx] = K[dataIdx];   
                }
            }
            __syncthreads();

            // matmul QK^T and softmax (reduces Bc elements into 32 registers using online softmax)
            float m_i = -INFINITY;
            float d_i = 0.0f;
            for (unsigned int i = 0; i < Bc; i += 32) {

                float S_val = 0.0f;
                for (unsigned int k = 0; k < D; ++k) {
                    float q_val = Q_i[ty * D + k];
                    float k_val = K_j[(tx + i) * D + k];
                    S_val += q_val * k_val;
                }
                // load result back to SMEM
                int dIdx_SMEM = ty * Bc + (tx + i);
                S_ij[dIdx_SMEM] = S_val;

                float m_prev = m_i;
                m_i = fmaxf(m_i, S_val);        // obtain new max
                d_i *= expf(m_prev - m_i);      // rescale old norm
                d_i += expf(S_val - m_i);       // add current contribution
            }

            // warp level online softmax
            for (unsigned int mirrorIdx = 1; mirrorIdx <= 16; mirrorIdx <<= 1) {
                float m_j = __shfl_xor_sync(FULL_MASK, m_i, mirrorIdx);     // obtain m_j
                float d_j = __shfl_xor_sync(FULL_MASK, d_i, mirrorIdx);     //obtain d_j

                float max = fmaxf(m_i, m_j);    // new max
                d_i *= expf(m_i - max);         // scale current norm
                d_i += d_j * expf(m_j - max);   // add contribution from other thread
                m_i = max;
            }

            // apply block softmax and load to S in GMEM
            for (unsigned int i = 0; i < Bc; i += 32) {
                int dIdx_GMEM = ty * N + (tx + i);
                int dIdx_SMEM = ty * Bc + (tx + i);
                float val = expf(S_ij[dIdx_SMEM] - m_i) / d_i;
                S[dIdx_GMEM] = val;
            }

            // calculate new global statistics (prev S chunk + running S chunk)
            float m_new = fmaxf(m_i, m_old);
            float d_new = d_old * expf(m_old - m_new) + d_i * expf(m_i - m_new);

            // PV matmul (next)

            // advance blocks
            K += Bc * D;        // K is not transposed so block offset is along rows.
            S += Bc;            // The next block load will shift S along columns.

            // store global stats as old stats
            m_old = m_new;
            d_old = d_new;

            __syncthreads();    // While a warp is busy with matmul, another warp can modify the very same SMEM.
            // Napkin proof: Take thread tx = 3, ty = 0. It loads the 3rd row of K_j. Take ty = 3, tx doesn't matter. 
            // If warp threads ty = 3 is faster than warp threads ty = 0, 
            // it will go and load the next K_j chunk 
            // while thread tx = 3, ty = 0 still needs the old chunk.
        }

    }
    
    template<const int Br, const int Bc, const int D>
    __global__ void PV_matmul (float *K, float *Q, float *V, float *O, const int N, const int d) {
        // P is (Br, Bc)
        // V is (Bc, d)
        // O is (Br, d)
        // O = PV
        // Every thread holds its responsible O elements inside registers. We have 32 threads per row and D
        // output elements per row of O. So each thread calculates and updates D/32 elements in a row of O.
        // The update of statistics are made by rescaling the previous O_reg result by the new max and norm, and then
        // adding the current (scaled by new global) block matmul result pv_sum.

        int rowIdx = blockIdx.x * Br;   // General block offset (also for the O matrix)
        Q += rowIdx * D;                // advance pointer rn. => GMEM -> SMEM load indexology is easier to read
        O += rowIdx * D;                // advance O pointer rn. => register -> GMEM load indexology is much simpler


        int tx = threadIdx.x % 32;      // ranges 0 to 31
        int ty = threadIdx.x / 32;      // ranges 0 to Br-1

        __shared__ float Q_i[Br * D];
        __shared__ float K_j[Bc * D];
        __shared__ float V_j[Bc * D];
        __shared__ float S_ij[Br * Bc];

        // previous block statistics
        float m_old = -INFINITY;
        float d_old = 0.0f;

        // thread result responsibilities
        const int ELEMENTS_PER_THREAD = D/32;
        float O_reg[ELEMENTS_PER_THREAD] = {0.0f};

        // load Q_i
        for (unsigned int i = 0; i < D; i += 32) {
            
            int dataIdx = ty * D + (tx + i);
            Q_i[dataIdx] = Q[dataIdx];
        }
        __syncthreads();

        // add block load offset loop here
        for (unsigned int block_load_offset = 0; block_load_offset < N; block_load_offset += Bc) {
            
            // load K_j and V_j: a bit more complicated
            for (unsigned int k = 0; k < Bc; k += Br) {

                int outer_dIdx = ty + k;
                for (unsigned int i = 0; i < D; i += 32) {
                    int inner_dIdx = tx + i;
                    int dataIdx = outer_dIdx * D + inner_dIdx;
                    
                    K_j[dataIdx] = K[dataIdx];
                    V_j[dataIdx] = V[dataIdx];  
                }
            }
            __syncthreads();

            // matmul QK^T and softmax (reduces Bc elements into 32 registers using online softmax)
            float m_i = -INFINITY;
            float d_i = 0.0f;
            for (unsigned int i = 0; i < Bc; i += 32) {

                float S_val = 0.0f;
                for (unsigned int k = 0; k < D; ++k) {
                    float q_val = Q_i[ty * D + k];
                    float k_val = K_j[(tx + i) * D + k];
                    S_val += q_val * k_val;
                }
                // load result back to SMEM
                int dIdx_SMEM = ty * Bc + (tx + i);
                S_ij[dIdx_SMEM] = S_val;

                float m_prev = m_i;
                m_i = fmaxf(m_i, S_val);        // obtain new max
                d_i *= expf(m_prev - m_i);      // rescale old norm
                d_i += expf(S_val - m_i);       // add current contribution
            }
            __syncthreads();

            // warp level online softmax
            for (unsigned int mirrorIdx = 1; mirrorIdx <= 16; mirrorIdx <<= 1) {
                float m_j = __shfl_xor_sync(FULL_MASK, m_i, mirrorIdx);     // obtain m_j
                float d_j = __shfl_xor_sync(FULL_MASK, d_i, mirrorIdx);     //obtain d_j

                float max = fmaxf(m_i, m_j);    // new max
                d_i *= expf(m_i - max);         // scale current norm
                d_i += d_j * expf(m_j - max);   // add contribution from other thread
                m_i = max;
            }

            // apply block softmax and load it to S_ij
            for (unsigned int i = 0; i < Bc; i += 32) {
                int dIdx_SMEM = ty * Bc + (tx + i);
                float unnorm_p_val = expf(S_ij[dIdx_SMEM] - m_i);
                S_ij[dIdx_SMEM] = unnorm_p_val;
            }
            __syncthreads();

            // calculate new global statistics (prev S chunk + running S chunk)
            float m_new = fmaxf(m_i, m_old);
            float d_new = d_old * expf(m_old - m_new) + d_i * expf(m_i - m_new);

            // PV matmul
            #pragma unroll
            for (unsigned int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                
                float pv_val = 0.0f;
                for (unsigned int k = 0; k < Bc; ++k) {
                    float p_val = S_ij[ty * Bc + k];
                    float v_val = V_j[k * D + (tx + i * 32)];
                    pv_val += p_val * v_val;
                }

                // update O_reg[i] result
                O_reg[i] *= (1/d_new) * d_old * expf(m_old - m_new);
                O_reg[i] += (1/d_new) * pv_val * expf(m_i - m_new);
            }
            __syncthreads();

            // advance blocks
            K += Bc * D;        // K is not transposed so block offset is along rows.
            V += Bc * D;        // V is offsetted across rows

            // store global stats as old stats
            m_old = m_new;
            d_old = d_new;
        }

        // load O_reg back to GMEM
        #pragma unroll
        for (unsigned int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
            int dIdx_GMEM = ty * D + (tx + i * 32);
            O[dIdx_GMEM] = O_reg[i];
        }

    }

    //
    // KERNEL WRAPPERS
    //
    void launch_QK_matmul (float *K, float *Q, float *S, const int N, const int d, cudaStream_t stream) {
        const int Br = 32;
        const int Bc = 32;
        // Make sure N and Bc are divisible by Br and 32, for simplicity.
        dim3 gridDim(CEIL_DIV(N, Br));
        dim3 blockDim(Br*32);

        switch(d) {
            case(32):
                QK_matmul<Br, Bc, 32><<<gridDim, blockDim, 0, stream>>>(K, Q, S, N, d); break;
            case(64):
                QK_matmul<Br, Bc, 64><<<gridDim, blockDim, 0, stream>>>(K, Q, S, N, d); break;
        }

        // Check for launch errors (like passing a CPU pointer!)
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            printf("Kernel Launch Error: %s\n", cudaGetErrorString(err));
    }

    void launch_S_softmax (float *K, float *Q, float *S, const int N, const int d, cudaStream_t stream) {
        const int Br = 32;
        const int Bc = 32;
        // Make sure N and Bc are divisible by Br and 32, for simplicity.
        dim3 gridDim(CEIL_DIV(N, Br));
        dim3 blockDim(Br*32);

        switch(d) {
            case(32):
                S_softmax<Br, Bc, 32><<<gridDim, blockDim, 0, stream>>>(K, Q, S, N, d); break;
            case(64):
                S_softmax<Br, Bc, 64><<<gridDim, blockDim, 0, stream>>>(K, Q, S, N, d); break;
        }

        // Check for launch errors (like passing a CPU pointer!)
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            printf("Kernel Launch Error: %s\n", cudaGetErrorString(err));
    }

    void launch_PV_matmul (float *K, float *Q, float *V, float *O, const int N, const int d, cudaStream_t stream) {
        const int Br = 32;
        const int Bc = 32;
        // Make sure N and Bc are divisible by Br and 32, for simplicity.
        dim3 gridDim(CEIL_DIV(N, Br));
        dim3 blockDim(Br*32);

        switch(d) {
            case(32):
                PV_matmul<Br, Bc, 32><<<gridDim, blockDim, 0, stream>>>(K, Q, V, O, N, d); break;
            case(64):
                PV_matmul<Br, Bc, 64><<<gridDim, blockDim, 0, stream>>>(K, Q, V, O, N, d); break;
        }

        // Check for launch errors (like passing a CPU pointer!)
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            printf("Kernel Launch Error: %s\n", cudaGetErrorString(err));
    }

}