#include <cstdio>
#include "benchmark.cuh"
#include "kernel.cuh"
#include "benchmark_common.cuh"


// -- CPU FUNCTIONS --
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
  *cpu_ref_time = (float)(cpu_stop - cpu_start); // Write to pointer
}

// -- VERIFY FUNCTIONS --
bool cpu_array_verify (float *gpu_res, float cpu_res, const int d) {
  if (std::abs(gpu_res[0] - cpu_res) > 1e-4) return false;
  return true;
}

bool cpu_rowSum_verify (float *gpu_res, float *cpu_res, const int N, const int d) {
  for (unsigned int i = 0; i < N; ++i) {
    float diff = std::abs(gpu_res[i * d] - cpu_res[i]);
    float relative_err = diff / std::abs(cpu_res[i]); // percentage error
    if (relative_err > 1e-5) {
      std::printf("Error seen at row %d\n", i);
      std::printf("GPU: %f\n", gpu_res[i * d]);
      std::printf("CPU: %f\n", cpu_res[i]);
      return false;
    }
  }
  return true;
}

// -- BENCHMARK FUNCTIONS --
void benchmark_rowSum (float *B, const int N, const int d, float cpu_ref_time, bool flush) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Get L2 cache size for flushing
  int device = 0;
  int l2_size = 0;
  float *d_F = nullptr;

  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaDeviceGetAttribute(&l2_size, cudaDevAttrL2CacheSize, device));
  size_t sizeF = l2_size * 2;

  // ALLOCATE-COPY 
  CUDA_CHECK(cudaMalloc((void **)&d_F, sizeF));
  CUDA_CHECK(cudaMemset((void *)d_F, 0 , sizeF));

  // WARM UP KERNEL
  for (int i = 0; i < 10; ++i) {
    test_rowSumXOR_v2(B, N, d);
  }

  // EXECUTION LOOP
  int iterations = 100;
  float ms = 0;
  for (int i = 0; i < iterations; ++i) {
    float partial_ms = 0;
    if (flush) {
      CUDA_CHECK(cudaMemsetAsync((void *)d_F, 0, sizeF));
      CUDA_CHECK(cudaDeviceSynchronize());
      CHECK_LAST_CUDA_ERROR();
    }

    cudaEventRecord(start);

    test_rowSumXOR_v2(B, N, d);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop); // Synchronize at cudaEvent_t stop
    cudaEventElapsedTime(&partial_ms, start, stop);
    ms += partial_ms;
  }

  // CALC AVG_TIME
  float avg_ms = ms / iterations;

  // BANDWIDTH CALCULATION
  // Formula: Bytes moved = (Read N * d + write N) * 4 Bytes, time = avg_ms
  double gb = (double)(N * d + N) * sizeof(float) / 1e9;
  double bandwidth = gb / (avg_ms / 1000.0);

  // PRINT RESULT
  std::printf("-- Benchmark Result --\n");
  std::printf("Average Time:  %.4f ms\n", avg_ms);
  std::printf("Throughput:    %.2f GB/s\n", bandwidth);
  std::printf("Speedup from CPU:  %.2fx\n", cpu_ref_time / avg_ms);
  std::cout << "Flush: " << std::boolalpha << flush << std::endl;

  // FREE FLUSH MEMORY
  CUDA_CHECK(cudaFree(d_F));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
}

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
    = std::bind(test_rowSumXOR_v2, B, N, d, sTtd::placeholders::_1);
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