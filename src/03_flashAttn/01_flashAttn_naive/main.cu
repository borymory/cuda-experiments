#include <cstdio>
#include "utils.cuh"
#include "kernel.cuh"
#include "benchmark.cuh"


// -- CPU FUNCTIONS --
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

void cpu_matmul (float *A, float *B, float *output, const int N, const int d, const int common_dim) {
  for (unsigned int row = 0; row < N; ++row) {
    for (unsigned int col = 0; col < d; ++col) {

      float sum = 0.0f;
      for (unsigned int k = 0; k < common_dim; ++k) {
        sum += A[row * d + k] * B[k * d + col];
      }
      output[row * d + col] = sum;
    }
  }
}

void cpu_transpose (float *input, const int N, const int d) {
  for (unsigned int i = 0; i < N; ++i) {
    for (unsigned int j = 0; j < i; ++j) {
      float placeholder;
      placeholder = input[j * d + i];
      input[j * d + i] = input[i * d + j];
      input[i * d + j] = placeholder;
    }
  }
}

void cpu_attention (float *K, float *Q, float *V, float *S, float *P, float *O, const int N, const int d) {
  cpu_transpose(K, N, d);
  cpu_matmul(Q, K, S, N, d, d);
  cpu_onlineSoftmax(S, P, N, N, nullptr);
  cpu_matmul(P, V, O, N, d, N);
}

// -- VERIFY FUNCTIONS --
// softmax can be verified element wise. Use validate func given in ./common/utils.cu

int main(void) {
  cudaStream_t stream;
  
  float *K;
  float *Q;
  float *V;
  float *O;
  float *K_cpu;
  float *Q_cpu;
  float *V_cpu;
  float *S_cpu; // INTERMEDIATE MATRIX (unfortunately materialized in CPU)
  float *P_cpu; // INTERMEDIATE MATRIX (unfortunately materialized in CPU)
  float *O_cpu;
  float cpu_time; // not used rn.

  const int N = 32;
  const int d = 32;


  // USE UNIFIED MEMORY - INITIALIATONS
  CUDA_CHECK(cudaMallocManaged(&K, N * d * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&Q, N * d * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&V, N * d * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&O, N * d * sizeof(float)));
  K_cpu = (float*)std::malloc(N * d * sizeof(float));
  Q_cpu = (float*)std::malloc(N * d * sizeof(float));
  V_cpu = (float*)std::malloc(N * d * sizeof(float));
  S_cpu = (float*)std::malloc(N * d * sizeof(float));
  P_cpu = (float*)std::malloc(N * d * sizeof(float));
  O_cpu = (float*)std::malloc(N * d * sizeof(float));
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
  FlashLab::initMatrix(K, N, d);
  FlashLab::initMatrix(Q, N, d);
  FlashLab::initMatrix(V, N, d);
  FlashLab::copyMatrix(K, K_cpu, N, d);
  FlashLab::copyMatrix(Q, Q_cpu, N, d);
  FlashLab::copyMatrix(V, V_cpu, N, d);
  cpu_attention(K_cpu, Q_cpu, V_cpu, S_cpu, P_cpu, O_cpu, N, d);
  FlashLab::flashAttn::naive::launch_flashAttn_fwd_v1(K, Q, V, O, N, d, stream);
  CUDA_CHECK(cudaDeviceSynchronize());
  if (FlashLab::validate(O, O_cpu, N * d)) std::printf("Succes!\n");  // VERIFY KERNEL

  // FREE MEMORY ALLOCATION
  CUDA_CHECK(cudaFree(K));
  CUDA_CHECK(cudaFree(Q));
  CUDA_CHECK(cudaFree(V));
  CUDA_CHECK(cudaFree(O));
  std::free(K_cpu);
  std::free(Q_cpu);
  std::free(V_cpu);
  std::free(S_cpu);
  std::free(P_cpu);
  std::free(O_cpu);

  // DESTROY STREAM
  CUDA_CHECK(cudaStreamDestroy(stream));

  return 0;
}