#include <stdio.h>
#include "utils.cuh"
#include "kernel.cuh"

///
/// TEMPLATE MAIN.CU FOLDER
///

// CPU CODE
void cpu_array_reduction (float *src, float cpu_res, const int d) {
  float tmpSum = 0.0f;
  for (uint i = 0, i < d; ++i) {
    tmpSum += src[i];
  }
  cpu_res = tmpSum;
}

bool cpu_array_verify (float *gpu_res, float cpu_res, const int d) {
  if (fabsf(gpu_res[0] - cpu_res) > 1e-4) return false;
  return true;
}

/// IF ELEMENT VISE VERIFICATION NEEDED, USE THE ONE GIVEN IN UTILS.CUH

int main(void) {
  
  float *B;
  float B_cpu = 0.0f;

  const int N = 256;
  const int d = 128;

  // USE UNIFIED MEMORY - INITIALIATONS
  cudaMallocManaged(&B, d * sizeof(float));

  initArray(B, d);

  // CREATE REFERANCE FOR CPU
  cpu_array_reduction(B, B_cpu, d);


  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  // RUN KERNEL
  test_vectorReduction_v2(B, d);

  // NOTE TIME STOP, ACTS AS SYNCHRONIZE
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Kernel Performance: %.2f milliseconds\n", milliseconds);

  // VERIFY/BENCHMARK KERNEL
  if (cpu_array_verify(B, B_cpu, d)) printf("Succes!\n");

  // FREE MEMORY ALLOCATION
  cudaFree(B);

  return 0;
}