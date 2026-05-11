#include "kernel.cuh"

__global__ void vectorReduction_v2 (float *A, const int d) {
    // 32 threads < d

    int tid = threadIdx.x;
    float threadVal = 0.0f;

    // Reduce the total size of array to 32 by storing every value inside registers
    for (uint offset = 0; offset < d; offset += 32) {
        // Failsafe if d is not a multiple of 32
        if (tid + offset < d)
            threadVal += A[tid + offset];
    }
    // Inside warp-level primitives __syncthreads() is not needed.
    // It is already handled by the hardware at __shfl_down_sync

    // Now A is reduced to the 32 threads
    // FULL_MASK is defined in kernel.cuh as 0xffffffff
    for (uint stride = 16; stride > 0; stride >>= 1) {
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

void test_vectorReduction_v2 (float *A, const int d) {
    dim3 blockDim(32);
    dim3 gridDim(1);

    vectorReduction_v2<<<gridDim, blockDim>>>(A, d);

    // Check for launch errors (like passing a CPU pointer!)
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Kernel Launch Error: %s\n", cudaGetErrorString(err));
}
