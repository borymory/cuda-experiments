#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "utils.cuh"
#include "kernel.cuh"

void cpu_reduction (float *src, float *dst, int N, int d) {
    for (int i = 0; i < N; i++) {
        float sum = 0;
        for (int j = 0; j < d; j++) sum += src[i * d + j];
        dst[i] = sum; // Ground truth
    }
}

int main(void) {
  
  float *B;
  float *B_ref;

  const int N = 256;
  const int d = 128;

  // USE UNIFIED MEMORY - INITIALIATONS
  cudaMallocManaged(&B, N * d * sizeof(float));
  cudaMallocHost(&B_ref, N * d * sizeof(float));

  initMatrix(B, N, d);

  // CREATE REFERANCE FOR CPU
  // memCpy(&B_ref, &B, N*d*sizeof(float)); not needed we just write on top of it

  // CALCULATE GROUND TRUTH - CPU
  cpu_reduction(B, B_ref, N, d);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  // RUN KERNEL
  test_matrixReduction_v1(B, N, d);

  // NOTE TIME STOP, ACTS AS SYNCHRONIZE
  cudaEventRecord(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Kernel Performance: %.2f milliseconds\n", milliseconds);

  // VERIFY/BENCHMARK KERNEL
  if (validate(B, B_ref, N*d)) printf("Succes!");

  // FREE MEMORY ALLOCATION
  cudaFree(B);
  cudaFree(B_ref);

  return 0;
}