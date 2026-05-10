#pragma once
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <math.h>
#include <cuda_runtime.h>

void test_vectorReduction_v1(float *A, const int d);

void test_matrixReduction_v1(float *B, const int N, const int d);