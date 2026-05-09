#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>

__global__ void vectorReduction_v1 (float *A, const int d) {
  // Assume A is 1xd.
  // Each block has 32 threads.
  // We launch (d/2)/32 many blocks.
  
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (uint stride = d>>1; stride > 0; stride>>=1) {
    __syncthreads();
    if (tid + stride < d)
      A[tid] += A[tid + stride];
  }
}

__global__ void matrixReduction_v1 (float *B, const int N, const int d) {
  // Assume A is Nxd.
  // Let each block have 32 threads.
  // Let each block calculate one row of A
  // d = 128. loop 1: each thread calculates 128/32 = 4 elements. Result, array sum of size 32
  // Loop 2: each thread does usual reduction
  // Even better: let each thread calculate d/32 * (1/2) = d/16 elements. In our example, 2. 
  // Then array size is threadcount * 2, which is suitable for the reduction algorithm.

 
  int rowIdx = blockIdx.x;
  int tid = threadIdx.x;

  // do the first reduction to array size * 2
  for (uint i = 0; i < d/32*1/2; offset += 32) {
    A[rowIdx * n + tid] += A[rowIdx * n + (tid + offset)];
  }

  // do the normal reduction algorithm with stride size 32
  for (uint stride = blockDim.x; stride > 0; stride>>=1) {
    __syncthreads();
    if (tid + stride < d)
      A[tid] += A[tid + stride];
  }
}