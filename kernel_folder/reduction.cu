#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>

__global__ void vectorreduction_v1(float *A, const int d) {
  // Assume A is 1xd.
  // Each block has 32 threads.
  // We launch (d/2)/32 many blocks.
  
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (uint stride = d>>1; stride > 0; stride>>=1) {
    if (tid + stride < d)
      A[tid] += A[tid + stride];
  }
}