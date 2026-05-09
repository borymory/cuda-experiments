#include <stdio.h>
#include "kernel_folder/kernels.cuh"
#include "utils.cuh"

void test_vectorreduction_v1(float *A, const int d) {
  
  dim3 blockDim(32);
  dim3 gridDim(d/32*2);

  vectorreduction_v1<<<gridDim, blockDim>>>(A, d);
}

void initArray(float *A, const int d) {
  for (uint i = 0; i < d; ++i) A[i] = (float)rand() / RAND_MAX;
}

void verify_vectorreduction_v1(float *ref_A, float *acc_A, const int d) {
  // acc_A: result
  // ref_A: CPU calculated result
  float tmpSum = 0.0f;
  for (uint i = 0; i < d; ++i) {
    tmpSum += ref_A[i];
  }
  if (tmpSum = acc_A[0]) printf("Hey it's accurate!\n");
  else printf("Hey it's NOT accurate!\n");
  printf("CPU Result: %f\n", tmpSum);
  printf("GPU Result: %f\n", acc_A[0]);
}