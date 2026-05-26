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

// Q is (N, d), K^T is (d, N), P is (N, N), V is (N, d)
void cpu_matmul (float *A, float *B, float *output, const int row_A, const int col_B, const int common_dim) {
  for (unsigned int row = 0; row < row_A; ++row) {
    for (unsigned int col = 0; col < col_B; ++col) {

      float sum = 0.0f;
      for (unsigned int k = 0; k < common_dim; ++k) {
        sum += A[row * common_dim + k] * B[k * col_B + col];
      }
      output[row * col_B + col] = sum;
    }
  }
}

// Q is (N, d), K is (N, d)
void cpu_matmul_withoutTranspose (float *A, float *B, float *output, const int row_A, const int row_B, const int common_dim) {
  for (unsigned int row = 0; row < row_A; ++row) {
    for (unsigned int col = 0; col < row_B; ++col) {

      float sum = 0.0f;
      for (unsigned int k = 0; k < common_dim; ++k) {
        sum += A[row * common_dim + k] * B[col * common_dim + k];
      }
      output[row * row_B + col] = sum;  // fun fact: I have written accidentally common_dim isntead of row_B
                                        // which led me to rewrite the very same kernel again and again...
    }
  }
}
// Q is (N,d), K is (N, d), V is (N, d)
void cpu_attention (float *K, float *Q, float *V, float *S, float *O, const int Q_row, const int K_row, const int common_dim) {
  cpu_matmul_withoutTranspose(Q, K, S, Q_row, K_row, common_dim);   // S is (Q_row, K_row), V is (K_row, common_dim)
  cpu_onlineSoftmax(S, S, Q_row, K_row, nullptr);
  cpu_matmul(S, V, O, Q_row, common_dim, K_row);
}

// Q is (N,d), K is (N, d)
void cpu_QK_matmul (float* K, float *Q, float *S, const int Q_row, const int K_row, const int common_dim) {
  cpu_matmul_withoutTranspose(Q, K, S, Q_row, K_row, common_dim);
}

void cpu_S_softmax (float* K, float *Q, float *S, const int Q_row, const int K_row, const int common_dim, const int Bc) {
  cpu_matmul_withoutTranspose(Q, K, S, Q_row, K_row, common_dim);
  
  // S is of size (Q_row, K_row)
  for (unsigned int row = 0; row < Q_row; ++row) {
    for (unsigned int col = 0; col < K_row; col += Bc) {
      
      float norm = 0.0f;
      float m_new = -INFINITY;
      float m_prev = -INFINITY;
      // do softmax on Bc element
      for (unsigned int k = 0; k < Bc; ++k) {
        float val = S[row * K_row + (col + k)];
        m_new = fmaxf(m_prev, val);   // calculate new max
        norm *= expf(m_prev - m_new); // rescale old sum
        norm += expf(val - m_new);    // add current contribution
        m_prev = m_new;
      }
      // store Bc results into S, simulating the flashAttn behaviour
      for (unsigned int k = 0; k < Bc; ++k) {
        float val = S[row * K_row + (col + k)];
        S[row * K_row + (col + k)] = expf(val - m_new) / norm;
      }
      
    }
  }
}

// -- VERIFY FUNCTIONS --
// These kernels can be verified element wise. Use validate func given in ./common/utils.cu

int main(void) {
  cudaStream_t stream;
  
  float *K;
  float *Q;
  float *V;
  float *O;

  float *K_cpu;
  float *Q_cpu;
  float *V_cpu;
  float *S_cpu;
  float *O_cpu;

  const int N = 32;
  const int d = 32;
  //const int Bc = 32;


  // USE UNIFIED MEMORY - INITIALIATONS
  CUDA_CHECK(cudaMallocManaged(&K, N * d * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&Q, N * d * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&V, N * d * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&O, N * d * sizeof(float)));
  K_cpu = (float*)std::malloc(N * d * sizeof(float));
  Q_cpu = (float*)std::malloc(N * d * sizeof(float));
  V_cpu = (float*)std::malloc(N * d * sizeof(float));
  S_cpu = (float*)std::malloc(N * N * sizeof(float));
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

  cpu_attention(K_cpu, Q_cpu, V_cpu, S_cpu, O_cpu, N, N, d);
  FlashLab::flashAttn::fundamentals::launch_PV_matmul(K, Q, V, O, N, d, stream);
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
  std::free(O_cpu);

  // DESTROY STREAM
  CUDA_CHECK(cudaStreamDestroy(stream));

  return 0;
}