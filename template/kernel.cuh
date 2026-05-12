#pragma once
#include <algorithm>
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define FULL_MASK 0xffffffffu // unsigned, safer in bit shifting

// let it hold function declarations. wrapper declarations are enough