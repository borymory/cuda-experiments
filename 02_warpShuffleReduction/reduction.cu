#include "kernel.cuh"
#include "utils.cuh"

// vectorReduction_v2:
// Uses __shfl_down_sync intrinsic for warp-level primitives

// vectorReductionXOR_v2:
// Uses __shfl_xor_sync intrinsic for warp-level primitives,
// leading to a full reduction. This option is better when all threads
// of the warp needs to have the total sum/reduced value.

// rowSumXOR_v2
// Uses __shfl_xor_sync to do a rowSum

// rowMaxXOR_v2
// Uses __shfl_xor_sync to do a rowMax


//
// KERNELS
//
__global__ void vectorReduction_v2 (float *A, const int d) {
    // d > 32

    int tid = threadIdx.x;
    float threadVal = 0.0f;

    // Reduce the total size of array to 32 by storing every value inside registers
    for (unsigned int offset = 0; offset < d; offset += 32) {
        // Failsafe if d is not a multiple of 32
        if (tid + offset < d)
            threadVal += A[tid + offset];
    }
    // Inside warp-level primitives __syncthreads() is not needed.
    // It is already handled by the hardware at __shfl_down_sync

    // Now A is reduced to the 32 threads
    // FULL_MASK is defined in kernel.cuh as 0xffffffff
    for (unsigned int stride = 16; stride > 0; stride >>= 1) {
        threadVal += __shfl_down_sync(FULL_MASK, threadVal, stride);
    }

    // write result back
    if (tid == 0) A[0] = threadVal;
}
// Common mistakes:
// using if (tid < stride) inside the shuffle reduction loop. Shuffle down has synchronization and needs all threads of the warp 
// specified by the mask to be present inside the call. If we used an if condition, then some threads wouldn't have made it
// which would cause the loop to stall indefinitely!
// Unsigned mask determines which threads of the warp will be participating in the shuffle

__global__ void vectorReductionXOR_v2 (float *A, const int d) {
    // Ghost sum enables d to be any positive number!

    int tid = threadIdx.x;
    float threadVal = 0.0f; 
    // IMPORTANT: Ghost Sum - If d = 8, threads 0-7 woud load it whereas threads 8-31 remain zero.
    // Thus, after the XOR shuffle, if we looked a any thread within the warp,
    // the contribution from threads 8-31 are virtually zero. Yet, all threads 
    // in the warp hold the reduced value!

    // Reduce data into registers:
    for (unsigned int offset = 0; offset < d; offset += 32) {
        // Fail safe if d is not a multiple of 32
        if (tid + offset < d)
            threadVal += A[tid + offset];
    }

    // Start XOR shuffle:
    for (unsigned int mirrorIdx = 1; mirrorIdx <= 16; mirrorIdx <<= 1) {
        threadVal += __shfl_xor_sync(FULL_MASK, threadVal, mirrorIdx);
    }

    // Write result back:
    if (tid == 0) A[0] = threadVal;
}

template<const int BN>
__global__ void rowSumXOR_v2 (float *B, const int N, const int d, float *B_out) {
    // We launch CEIL_DIV(N, BN) many blocks
    // We have BN * 32 many threads per block

    int rowIdx = blockIdx.x * BN;

    // Offset each block to a row
    B += rowIdx * d;

    // Assume each block has BN many warps
    int tx = threadIdx.x % 32;  // Inner Col of block (same as LaneID)
    int ty = threadIdx.x / 32;  // Inner Row of block

    float threadVal = 0.0f;
    // If N is not a multiple of 32
    if (rowIdx + ty < N) {
        // Reduce data into registers
        for (unsigned int offset = 0; offset < d; offset += 32) {
            // If d is not a multiple of 32
            if (tx + offset < d)
                threadVal += B[ty * d + (tx + offset)];
        }

        // Start XOR shuffle:
        for (unsigned int mirrorIdx = 1; mirrorIdx <= 16; mirrorIdx <<= 1) {
            threadVal += __shfl_xor_sync(FULL_MASK, threadVal, mirrorIdx);
        }

        // Write result back
        if (tx == 0) B[ty * d] = threadVal;
    }
}

//
// KERNEL WRAPPERS
//
void test_vectorReduction_v2 (float *A, const int d) {
    dim3 blockDim(32);
    dim3 gridDim(1);

    vectorReduction_v2<<<gridDim, blockDim>>>(A, d);

    // Check for launch errors (like passing a CPU pointer!)
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Kernel Launch Error: %s\n", cudaGetErrorString(err));
}

void test_vectorReductionXOR_v2 (float *A, const int d) {
    dim3 blockDim(32); // Working strictly with warps
    dim3 gridDim(1);

    vectorReductionXOR_v2<<<gridDim, blockDim>>>(A, d);

    // Check for launch errors (like passing a CPU pointer!)
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Kernel Launch Error: %s\n", cudaGetErrorString(err));
}

void test_rowSumXOR_v2 (float *B, const int N, const int d, float *B_out) {
    const int BN = 8;
    dim3 blockDim(BN * 32);
    dim3 gridDim(CEIL_DIV(N, BN));

    rowSumXOR_v2<BN><<<gridDim, blockDim>>>(B, N, d, B_out);

    // Check for launch errors (like passing a CPU pointer!)
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Kernel Launch Error: %s\n", cudaGetErrorString(err));
}