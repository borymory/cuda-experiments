#include <cstdio>
#include "utils.cuh"
#include "kernel.cuh"

// CPU CODE - declare float cpu_res and pass it as &cpu_res into argument inside main func
void cpu_array_reduction (float *src, float *cpu_res, const int d) {
  float tmpSum = 0.0f;
  for (unsigned int i = 0; i < d; ++i) {
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

void cpu_rowSum (float *src, float *cpu_res, const int N, const int d, float *cpu_ref_time) {
  double cpu_start = get_time_ms();
  for (unsigned int i = 0; i < N; ++i) {

    float rowResult = 0.0f;
    for (unsigned int j = 0; j < d; ++j)
      rowResult += src[i * d + j];
    cpu_res[i] = rowResult;
  }
  double cpu_stop = get_time_ms();
  *cpu_ref_time = (float)(cpu_stop - cpu_start);
}

bool cpu_array_verify (float *gpu_res, float cpu_res, const int d) {
  if (std::abs(gpu_res[0] - cpu_res) > 1e-4) return false;
  return true;
}

bool cpu_rowSum_verify (float *gpu_res, float *cpu_res, const int N, const int d) {
  for (unsigned int i = 0; i < N; ++i) {
    if (std::abs(gpu_res[i * d] - cpu_res[i]) > 1e-4) return false;
  }
  return true;
}

//
//-- BENCHMARK --
//

void benchmark_rowSum (float *B, const int N, const int d, float *B_out, float cpu_ref_time) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // WARM UP KERNEL
  for (int i = 0; i < 10; ++i) {
    test_rowSumXOR_v2(B, N, d, B_out);
  }

  // EXECUTION LOOP
  int iterations = 100;
  cudaEventRecord(start);
  for (int i = 0; i < iterations; ++i) {
    test_rowSumXOR_v2(B, N, d, B_out);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop); // Acts as synchronize

  // NOTE TIME STOP
  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  float avg_ms = ms / iterations;

  // BANDWIDTH CALCULATION
  // Formula: Bytes moved = (Read N * d + write N) * 4 Bytes, time = avg_ms
  double gb = (double)(N * d + N) * sizeof(float) / 1e9;
  double bandwidth = gb / (avg_ms / 1000.0);

  // PRINT RESULT
  std::printf("-- Benchmark Result --\n");
  std::printf("Average Time:  %.4f ms\n", avg_ms);
  std::printf("Throughput:    %.2f Gb/s\n", bandwidth);
  std::printf("Speedup from CPU:  %.2fx\n", cpu_ref_time / avg_ms);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

/// IF ELEMENT VISE VERIFICATION NEEDED, USE THE ONE GIVEN IN UTILS.CUH

int main(void) {
  
  float *B;
  float *B_out;
  float *B_cpu;
  float cpu_ref_time;

  const int N = 16384;
  const int d = 1024;

  // USE UNIFIED MEMORY - INITIALIATONS
  cudaMallocManaged(&B, N * d * sizeof(float));
  cudaMallocManaged(&B_out, N * d * sizeof(float));
  B_cpu = (float*)std::malloc(N * sizeof(float));

  initMatrix(B, N, d);

  // CREATE REFERANCE FOR CPU - TIME IT
  cpu_rowSum(B, B_cpu, N, d, &cpu_ref_time);

  // RUN KERNEL

  // BENCHMARK/VERIFY KERNEL
  benchmark_rowSum(B, N, d, B_out, cpu_ref_time);
  if (cpu_rowSum_verify(B, B_cpu, N, d)) std::printf("Succes!\n");

  // FREE MEMORY ALLOCATION
  cudaFree(B);
  cudaFree(B_out);
  std::free(B_cpu);

  return 0;
}