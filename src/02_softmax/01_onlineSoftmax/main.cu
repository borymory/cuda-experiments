#include <cstdio>
#include "utils.cuh"
#include "kernel.cuh"
#include "benchmark.cuh"


// -- CPU FUNCTIONS --
void cpu_safeSoftmax (float *input, float *output, int N, int d) {
  for (unsigned int i = 0; i < N; ++i) {

    // find max
    float m = -INFINITY;
    for (unsigned int k = 0; k < d; ++k) {
      m = fmaxf(m, input[i * d + k]);
    }

    // find sum (safe)
    float sum = 0.0f;
    for (unsigned int k = 0; k < d; ++k) {
      sum += expf(input[i * d + k] - m);
    }

    // calculate softmax
    for (unsigned int k = 0; k < d; ++k) {
      output[i * d + k] = expf(input[i * d + k] - m) / sum;
    }
  }

  // 4 memory access per vector element. We have N*d element. Total of 4*N*d many mem access. 
}

// Pass nullptr if you don't want to time it
void cpu_onlineSoftmax (float *input, float *output, const int N, const int d, float *time_cpu) {
  double cpu_start;
  if (time_cpu != nullptr) cpu_start = FlashLab::get_time_ms();
  for (unsigned int i = 0; i < N; ++i) {

    // running normalizer and max
    float m_new = -INFINITY;
    float m_prev = -INFINITY;
    float norm = 0.0f;
    for (unsigned int k = 0; k < d; ++k) {
      m_new = fmaxf(m_prev, input[i * d + k]);

      norm *= expf(m_prev - m_new);             // adjust to new max
      norm += expf(input[i * d + k] - m_new);   // add new sum
      m_prev = m_new;                           // set new max to prev
    }
    // don't forget that we cannot 'just hold onto max' until the final iteration. We have to constantly subtract it while doing
    // our sums to make sure that overflow/underflow does not occur!


    // calculate final values
    for (unsigned int k = 0; k < d; ++k) {
      output[i * d + k] = expf(input[i * d + k] - m_new) / norm;
    }
  }
  if (time_cpu != nullptr) *time_cpu = static_cast<float>(FlashLab::get_time_ms() - cpu_start);
}

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
  const int BN = 8;


  // USE UNIFIED MEMORY - INITIALIATONS
  CUDA_CHECK(cudaMallocManaged(&input, N * d * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&output, N * d * sizeof(float)));
  output_cpu = (float*)std::malloc(N * d * sizeof(float));
  CUDA_CHECK(cudaStreamCreate(&stream));

  // -- BENCHMARK STATISTICS --
  constexpr size_t num_repeats = 10000;
  constexpr size_t num_warmups = 1000;
  size_t bytes_moved = static_cast<size_t>(2 * N * d) * sizeof(float);

  // -- BENCHMARK KERNEL RUN --
  FlashLab::initMatrix(input, N, d);                              // Init Matrix
  cpu_onlineSoftmax(input, output_cpu, N, d, &cpu_time);          // Store CPU Time
  std::function<void(cudaStream_t)> launch_kernel
    = std::bind(FlashLab::Softmax::launch_softmax_v2<float, BN>, input, output, N, d, std::placeholders::_1);
  FlashLab::Benchmark::benchmark_kernel(launch_kernel, stream, bytes_moved, cpu_time, num_repeats, num_warmups, true);

  // -- VERIFY KERNEL RUN --
  FlashLab::initMatrix(input, N, d);                              // Init Matrix
  cpu_onlineSoftmax(input, output_cpu, N, d, nullptr);            // Store CPU Result
  FlashLab::Softmax::launch_softmax_v2<float, BN>(input, output, N, d, stream); // Store GPU Result
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