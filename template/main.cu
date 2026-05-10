#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "utils.cuh"
#include "kernel.cuh"

///
/// TEMPLATE MAIN.CU FOLDER
///

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
  memCpy(&B_ref, &B, N*d*sizeof(float));


  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  // RUN KERNEL
  some_kernel_call_function(B, N, d);

  // NOTE TIME STOP, ACTS AS SYNCHRONIZE
  cudaEventRecord(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Kernel Performance: %.2f milliseconds\n", milliseconds);

  // VERIFY/BENCHMARK KERNEL
  verifybenchmark_kernel(B_ref, B, N, d);

  // FREE MEMORY ALLOCATION
  cudaFree(B);
  cudaFree(B_ref);

  return 0;
}