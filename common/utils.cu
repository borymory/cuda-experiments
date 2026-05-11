#include "utils.cuh"

void initArray(float *A, const int d) {
  for (uint i = 0; i < d; ++i) A[i] = (float)rand() / RAND_MAX;
}

void initMatrix(float *B, const int N, const int d) {
  for (uint i = 0; i < N; i++) {
    for (uint j = 0; j < d; ++j) B[i * d + j] = (float)rand() / RAND_MAX;
  }
}

void copyArray (float *src, float *dst, const int d) {
  for (uint i = 0; i < d; ++i) dst[i] = src[i];
}

void copyMatrix(float *src, float *dst, const int N, const int d) {
  for (uint i = 0; i < N; i++) {
    for (uint j = 0; j < d; ++j) dst[i * d + j] = src[i * d + j];
  }
}

// generic verifier
bool validate(float *gpu_res, float *cpu_res, int size) {
    for (int i = 0; i < size; i++) {
        if (fabsf(gpu_res[i] - cpu_res[i]) > 1e-4) return false;
    }
    return true;
}