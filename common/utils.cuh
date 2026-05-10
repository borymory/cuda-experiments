#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

#define CUDA_CHECK(expr_to_check) do {            \
    cudaError_t result  = expr_to_check;          \
    if(result != cudaSuccess)                     \
    {                                             \
        fprintf(stderr,                           \
                "CUDA Runtime Error: %s:%i:%d = %s\n", \
                __FILE__,                         \
                __LINE__,                         \
                result,\
                cudaGetErrorString(result));      \
    }                                             \
} while(0)

// array initializer of size d
void initArray(float *A, const int d);

void initMatrix(float *B, const int N, const int d);

void copyArray (float *src, float *dst, const int d);

void copyMatrix(float *src, float *dst, const int N, const int d);


// verify element wise
bool validate(float *gpu_res, float *cpu_res, int size);