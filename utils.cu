#include <stdio.h>
#include "kernel_folder/kernels.cuh"
#include "utils.cuh"

void test_vectorReduction_v1(float *A, const int d) {
  
  dim3 blockDim(32);
  dim3 gridDim(d/32*2);

  vectorReduction_v1<<<gridDim, blockDim>>>(A, d);
}

void test_matrixReduction_v1(float *B, const int N, const int d) {
  
  dim3 blockDim(32);
  dim3 gridDim(N);

  matrixReduction_v1<<<gridDim, blockDim>>>(B, N, d);
}

void initArray(float *A, const int d) {
  for (uint i = 0; i < d; ++i) A[i] = (float)rand() / RAND_MAX;
}

void initMatrix(float *B, const int N, const int d) {
  for (uint i = 0; i < N; i++) {
    for (uint j = 0; j < d; ++j) B[i * d + j] = (float)rand() / RAND_MAX;
  }
}

void copyArray (float *src, float *dst, const int d) {
  for (uint i = 0; i < d; ++i) dst[i] = src[i];
}

void copyMatrix(float *src, float *dst, const int N, const int d) {
  for (uint i = 0; i < N; i++) {
    for (uint j = 0; j < d; ++j) dst[i * d + j] = src[i * d + j];
  }
}

void verify_vectorReduction_v1(float *ref_A, float *acc_A, const int d) {
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

void verify_matrixReduction_v1(float *ref_B, float *acc_B, const int N, const int d) {
  // acc_B: result
  // ref_B: CPU calculated result
  bool errorSeen = false;
  for (uint row = 0; row < N; ++row) {
    float rowSum = 0.0f;
    for (uint i = 0; i < d; ++i) {
      rowSum += ref_B[row * d + i];
    }
    if (rowSum != acc_B[row * d]) {
      errorSeen = true;
      printf("Problem seen at row: %d\n", row);
      printf("CPU Result: %f\n", rowSum);
      printf("GPU Result: %f\n", acc_B[row * d]);
    }
  }
  if (errorSeen == false) printf("Hey it's accurate!");
}