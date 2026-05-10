#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>

__global__ void vectorReduction_v1 (float *A, const int d) {
  // A is 1xd.
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
  // B is Nxd.
  // Let each block have 32 threads.

  int rowIdx = blockIdx.x;
  int tid = threadIdx.x;
  
  // we reduce the total size to 32 and then load to shared memory
  __shared__ float Bs[32];
  float tmpSum = 0.0f;
  for (uint offset = 0; offset < d; offset += 32) {
    if (tid + offset < d)
      tmpSum += B[rowIdx * d + (tid + offset)];
  }
  Bs[tid] = tmpSum;
  __syncthreads(); // make sure all threads have written the values to SMEM before reading.

  // then follows a simple reduction in the shared memory: we use stride = blockDim.x>>1 since element count and array size match
  for (uint stride = blockDim.x>>1; stride > 0; stride>>=1) {
    if (tid < stride)
      Bs[tid] += Bs[tid + stride];
    __syncthreads(); // make sure the all threads have written their sum value
  }

  if (tid == 0)
    B[rowIdx * d] = Bs[0];
}