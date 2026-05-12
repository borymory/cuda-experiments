#include <cstdio>
#include "utils.cuh"
#include "kernel.cuh"

// CPU CODE - declare float cpu_res and pass it as &cpu_res into argument inside main func
void cpu_array_reduction (float *src, float *cpu_res, const int d) {
  float tmpSum = 0.0f;
  for (uint i = 0; i < d; ++i) {
    tmpSum += src[i];
  }
  *cpu_res = tmpSum; // write to pointer
}
// A short note to myself about float pointers and floats: float *B is a float pointer
// and thus the variable B itself is an adress. On the other hand, float B_cpu is only
// a vairable that holds a value. To create the memory adress for that, 
// we pass in &B_cpu into the above argument.
// *cpu_res = tmpSum: cpu_res is a pointer. tmpSum is a value. If we just did
// cpu_res = tmpSum, then the pointer (literal adress) is changed.
// Instead we add a * (dereference operator) which makes the computer not look at the pointer but
// to the value inside that pointer and modify that to tmpSum

void cpu_rowSum (float *src, float *cpu_res, const int N, const int d) {

  for (unsigned int i = 0; i < N; ++i) {

    float rowResult = 0.0f;
    for (unsigned int j = 0; j < d; ++j)
      rowResult += src[i * d + j];
    cpu_res[i] = rowResult;
  }

}

bool cpu_array_verify (float *gpu_res, float cpu_res, const int d) {
  if (std::fabsf(gpu_res[0] - cpu_res) > 1e-4) return false;
  return true;
}

bool cpu_rowSum_verify (float *gpu_res, float *cpu_res, const int N, const int d) {
  for (unsigned int i = 0; i < N; ++i) {
    if (std::fabsf(gpu_res[i * d] - cpu_res[i]) > 1e-4) return false;
  }
  return true;
}

/// IF ELEMENT VISE VERIFICATION NEEDED, USE THE ONE GIVEN IN UTILS.CUH

int main(void) {
  
  float *B;
  float *B_cpu;

  const int N = 256;
  const int d = 128;

  // USE UNIFIED MEMORY - INITIALIATONS
  cudaMallocManaged(&B, N * d * sizeof(float));
  B_cpu = (float*)std::malloc(N * sizeof(float));

  initMatrix(B, N, d);

  // CREATE REFERANCE FOR CPU
  cpu_rowSum(B, B_cpu, N, d);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  // RUN KERNEL
  test_rowSumXOR_v2(B, N, d);

  // NOTE TIME STOP, ACTS AS SYNCHRONIZE
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::printf("Kernel Performance: %.2f milliseconds\n", milliseconds);

  // VERIFY/BENCHMARK KERNEL
  if (cpu_rowSum_verify(B, B_cpu, N, d)) std::printf("Succes!\n");

  // FREE MEMORY ALLOCATION
  cudaFree(B);
  std::free(B_cpu);

  return 0;
}