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

void cpu_onlineSoftmax (float *input, float *output, const int N, const int d) {

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

// -- VERIFY FUNCTIONS --


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
  FlashLab::initMatrix(B, N, d);  // INIT MATRIX
  cpu_rowSum(B, B_cpu, N, d, &cpu_ref_time); // WRITE GET CPU TIME

  // -- BENCHMARK KERNEL RUN --
  std::function<void(cudaStream_t)> launch_kernel 
    = std::bind(FlashLab::Reduction::test_rowSumXOR_v2, B, N, d, std::placeholders::_1);
  FlashLab::Benchmark::benchmark_kernel(launch_kernel, stream, bytes_moved, cpu_ref_time, num_repeats, num_warmups, true);

  // -- VERIFY KERNEL RUN --
  FlashLab::initMatrix(B, N, d);  // INIT MATRIX
  cpu_rowSum(B, B_cpu, N, d, &cpu_ref_time); // GET CPU RESULT
  FlashLab::Reduction::test_rowSumXOR_v2(B, N, d, stream); // GET GPU RESULT
  CUDA_CHECK(cudaDeviceSynchronize());
  if (cpu_rowSum_verify(B, B_cpu, N, d)) std::printf("Succes!\n");  // VERIFY KERNEL

  // FREE MEMORY ALLOCATION
  CUDA_CHECK(cudaFree(B));
  std::free(B_cpu);

  // DESTROY STREAM
  CUDA_CHECK(cudaStreamDestroy(stream));

  return 0;
}