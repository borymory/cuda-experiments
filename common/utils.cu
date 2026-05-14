#include "utils.cuh"

double get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    // Convert seconds and microseconds to a single millisecond value
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}

void initArray(float *A, const int d) {
  for (unsigned int i = 0; i < d; ++i) A[i] = (float)std::rand() / RAND_MAX;
}

void initMatrix(float *B, const int N, const int d) {
  for (unsigned int i = 0; i < N; i++) {
    for (unsigned int j = 0; j < d; ++j) B[i * d + j] = (float)std::rand() / RAND_MAX;
  }
}

void copyArray (float *src, float *dst, const int d) {
  for (unsigned int i = 0; i < d; ++i) dst[i] = src[i];
}

void copyMatrix(float *src, float *dst, const int N, const int d) {
  for (unsigned int i = 0; i < N; i++) {
    for (unsigned int j = 0; j < d; ++j) dst[i * d + j] = src[i * d + j];
  }
}

// generic verifier
bool validate(float *gpu_res, float *cpu_res, int size) {
    for (unsigned int i = 0; i < size; i++) {
        if (std::abs(gpu_res[i] - cpu_res[i]) > 1e-4) return false;
    }
    return true;
}

void checkLast(const char* const file, const int line)
    {
        cudaError_t const err{cudaGetLastError()};
        if (err != cudaSuccess)
        {
            std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                    << std::endl;
            std::cerr << cudaGetErrorString(err) << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }