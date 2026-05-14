#include <cstdio>
#include "utils.cuh"
#include "kernel.cuh"

void cpu_reduction (float *src, float *dst, const int N, const int d) {
    for (unsigned int i = 0; i < N; i++) {
        float sum = 0;
        for (unsigned int j = 0; j < d; j++) sum += src[i * d + j];
        // Sum is stored at first column of each row, just like the kernel
        dst[i * d] = sum; // Ground truth
    }
}

// VERIFY FIRST COLUMN OF EACH ROW OF CPU AND GPU
bool cpu_verify (float *gpu_res, float *cpu_res, const int N, const int d) {
  for (unsigned int i = 0; i < N; i++) {
    if (std::abs(gpu_res[i * d] - cpu_res[i * d]) > 1e-4) return false;
  }
  return true;
}

int main(void) {
  
  float *B;
  float *B_ref;

  const int N = 256;
  const int d = 128;

  // USE UNIFIED MEMORY - INITIALIATONS
  cudaMallocManaged(&B, N * d * sizeof(float));
  B_ref = (float*)std::malloc(N * d* sizeof(float));

  initMatrix(B, N, d);

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
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::sprintf("Kernel Performance: %.2f milliseconds\n", milliseconds);

  // VERIFY/BENCHMARK KERNEL
  if (cpu_verify(B, B_ref, N, d)) std::printf("Succes!\n");

  // FREE MEMORY ALLOCATION
  cudaFree(B);
  std::free(B_ref);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}