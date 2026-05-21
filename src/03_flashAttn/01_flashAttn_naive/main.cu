#include <cstdio>
#include "utils.cuh"
#include "kernel.cuh"
#include "benchmark.cuh"


// -- CPU FUNCTIONS --

// -- VERIFY FUNCTIONS --
// softmax can be verified element wise. Use validate func given in ./common/utils.cu

int main(void) {
  cudaStream_t stream;
  
  float *input;
  float *output;
  float *output_cpu;
  float cpu_time;

  const int N = 16384;
  const int d = 128;


  // USE UNIFIED MEMORY - INITIALIATONS
  CUDA_CHECK(cudaMallocManaged(&input, N * d * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&output, N * d * sizeof(float)));
  output_cpu = (float*)std::malloc(N * d * sizeof(float));
  CUDA_CHECK(cudaStreamCreate(&stream));

  // -- BENCHMARK STATISTICS --
  //constexpr size_t num_repeats = 10000;
  //constexpr size_t num_warmups = 1000;
  //size_t bytes_moved = static_cast<size_t>(2 * N * d) * sizeof(float);

  // -- BENCHMARK KERNEL RUN --
  //FlashLab::initMatrix(input, N, d);                              // Init Matrix
  //cpu_onlineSoftmax(input, output_cpu, N, d, &cpu_time);          // Store CPU Time
  //std::function<void(cudaStream_t)> launch_kernel
  //  = std::bind(FlashLab::Softmax::launch_softmax_v2, type, BN, input, output, N, d, std::placeholders::_1);
  //FlashLab::Benchmark::benchmark_kernel(launch_kernel, stream, bytes_moved, cpu_time, num_repeats, num_warmups, true);

  // -- VERIFY KERNEL RUN --
  FlashLab::initMatrix(input, N, d);                                  // Init Matrix
  cpu_onlineSoftmax(input, output_cpu, N, d, nullptr);                // Store CPU Result
  FlashLab::Softmax::launch_softmax_v2(type, BN, input, output, N, d, stream);  // Store GPU Result
  CUDA_CHECK(cudaDeviceSynchronize());
  if (FlashLab::validate(output, output_cpu, N * d)) std::printf("Succes!\n");  // VERIFY KERNEL

  // FREE MEMORY ALLOCATION
  CUDA_CHECK(cudaFree(input));
  CUDA_CHECK(cudaFree(output));
  std::free(output_cpu);

  // DESTROY STREAM
  CUDA_CHECK(cudaStreamDestroy(stream));

  return 0;
}