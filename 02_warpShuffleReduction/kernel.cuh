#pragma once
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <math.h>
#include <cuda_runtime.h>
#define FULL_MASK 0xffffffff

void test_vectorReduction_v2 (float *A, const int d);

void test_vectorReductionXOR_v2 (float *A, const int d);