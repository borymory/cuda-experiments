#include <cstdio>
#include "benchmark.cuh"
#include "kernel.cuh"
#include "benchmark_common.cuh"

//
// -- CPU FUNCTIONS --
//
// CPU CODE - declare float cpu_res and pass it as &cpu_res into argument inside main func

//
// -- VERIFY FUNCTIONS --
//
/// IF ELEMENT VISE VERIFICATION NEEDED, USE THE ONE GIVEN IN UTILS.CUH

int main(void) {
  cudaStream_t stream;
  
  float *B;
  float *B_cpu;
  float cpu_ref_time;

  const int N = 16384;
  const int d = 1024;

  size_t bytes_moved = (double)(N * d + N) * sizeof(float);

  // USE UNIFIED MEMORY - INITIALIATONS
  CUDA_CHECK(cudaMallocManaged(&B, N * d * sizeof(float)));
  B_cpu = (float*)std::malloc(N * sizeof(float));
  CUDA_CHECK(cudaStreamCreate(&stream));

  // -- BENCHMARK STATISTICS --
  constexpr size_t num_repeats = 10000;
  constexpr size_t num_warmups = 1000;
  initMatrix(B, N, d);  // INIT MATRIX
  cpu_rowSum(B, B_cpu, N, d, &cpu_ref_time); // WRITE GET CPU TIME

  // -- BENCHMARK KERNEL RUN --
  std::function<void(cudaStream_t)> launch_kernel 
    = std::bind(test_rowSumXOR_v2, B, N, d, std::placeholders::_1);
  benchmark_kernel(launch_kernel, stream, bytes_moved,  cpu_ref_time, num_repeats, num_warmups, true);

  // -- VERIFY KERNEL RUN --
  initMatrix(B, N, d);  // INIT MATRIX
  cpu_rowSum(B, B_cpu, N, d, &cpu_ref_time); // GET CPU RESULT
  test_rowSumXOR_v2(B, N, d, stream); // GET GPU RESULT
  CUDA_CHECK(cudaDeviceSynchronize());
  if (cpu_rowSum_verify(B, B_cpu, N, d)) std::printf("Succes!\n");  // VERIFY KERNEL

  // FREE MEMORY ALLOCATION
  CUDA_CHECK(cudaFree(B));
  std::free(B_cpu);

  // DESTROY STREAM
  CUDA_CHECK(cudaStreamDestroy(stream));

  return 0;
}