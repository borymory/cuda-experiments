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
    

    //
    // KERNEL WRAPPERS
    //
    void launch_QK_matmul (float *K, float *Q, float *S, const int N, const int d, cudaStream_t stream) {
        const int Br = 32;
        const int Bc = 32;
        // make sure N is 128 or something
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

}