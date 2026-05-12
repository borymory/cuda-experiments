#include <cstdio>
#include "utils.cuh"
#include "kernel.cuh"

///
/// TEMPLATE MAIN.CU FOLDER
///

/// PLACE YOUR KERNEL SPECIFIC VERIFICATION CODE HERE
/// IF ELEMENT VISE VERIFICATION NEEDED, USE THE ONE GIVEN IN UTILS.CUH

int main(void) {
  
  float *B;
  float *B_cpu;

  const int N = 256;
  const int d = 128;

  // USE UNIFIED MEMORY - INITIALIATONS
  cudaMallocManaged(&B, N * d * sizeof(float));
  B_cpu = (float*)std::malloc(N * d * sizeof(float));

  initMatrix(B, N, d);

  // CREATE REFERANCE FOR CPU
  memCpy(&B_cpu, &B, N*d*sizeof(float));


  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  // RUN KERNEL
  some_kernel_call_function(B, N, d);

  // NOTE TIME STOP, ACTS AS SYNCHRONIZE
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::printf("Kernel Performance: %.2f milliseconds\n", milliseconds);

  // VERIFY/BENCHMARK KERNEL
  verifybenchmark_kernel(B, B_cpu, N, d);

  // FREE MEMORY ALLOCATION
  cudaFree(B);
  std::free(B_cpu);

  return 0;
}