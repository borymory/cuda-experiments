#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>    // For std::cerr, std::endl
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <unistd.h>
#include <sys/time.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

#define CHECK_LAST_CUDA_ERROR() FlashLab::checkLast(__FILE__, __LINE__)

#define CUDA_CHECK(expr_to_check) do {            \
    cudaError_t result  = expr_to_check;          \
    if(result != cudaSuccess)                     \
    {                                             \
        std::fprintf(stderr,                           \
                "CUDA Runtime Error: %s:%i:%d = %s\n", \
                __FILE__,                         \
                __LINE__,                         \
                result,\
                cudaGetErrorString(result));      \
    }                                             \
} while(0)

namespace FlashLab {

    double get_time_ms();

    // array initializer of size d
    void initArray(float *A, const int d);

    void initMatrix(float *B, const int N, const int d);

    void copyArray (float *src, float *dst, const int d);

    void copyMatrix(float *src, float *dst, const int N, const int d);

    // verify element wise
    bool validate(float *gpu_res, float *cpu_res, int size);

    void checkLast(const char* const file, const int line);

}