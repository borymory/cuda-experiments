#include <stdio.h>
#include "utils.cuh"
#include "kernel.cuh"

///
/// TEMPLATE MAIN.CU FOLDER
///

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

bool cpu_array_verify (float *gpu_res, float cpu_res, const int d) {
  if (fabsf(gpu_res[0] - cpu_res) > 1e-4) return false;
  return true;
}

/// IF ELEMENT VISE VERIFICATION NEEDED, USE THE ONE GIVEN IN UTILS.CUH

int main(void) {
  
  float *B;
  float B_cpu;

  // const int N = 256;
  const int d = 128;

  // USE UNIFIED MEMORY - INITIALIATONS
  cudaMallocManaged(&B, d * sizeof(float));

  initArray(B, d);

  // CREATE REFERANCE FOR CPU
  cpu_array_reduction(B, &B_cpu, d); // pass adress of B_cpu


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