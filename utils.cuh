#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

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

void copyArray (const float *src, float *dst, const int d);

void copyMatrix(const float *src, float *dst, const int N, const int d);


// test kernels
void test_vectorReduction_v1(float *A, const int d);

void test_matrixReduction_v1(float *B, const int N, const int d)



// verify kernels
void verify_vectorReduction_v1(float *ref_A, float *acc_A, const int d);

void verify_matrixReduction_v1(float *ref_B, float *acc_B, const int N, const int d);