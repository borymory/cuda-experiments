#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "utils.cuh"

int main(void) {

  // pointer for Host memory
  float *A;
  float *A_ref;
  // pointer for Device memory
  float *devA;

  const int d = 128;

  // allocate host memory
  cudaMallocHost(&A, d*sizeof(float));
  cudaMallocHost(&A_ref, d*sizeof(float));

  // initialize vector
  initArray(A, d);
  for (int i = 0; i < d; i++) A_ref[i] = A[i];
  
  // allocate memory on Device
  cudaMalloc(&devA, d*sizeof(float));

  // copy from CPU to GPU
  cudaMemcpy(devA, A, d*sizeof(float), cudaMemcpyDefault);

  test_vectorreduction_v1(A, d);

  // wait for kernel to be done
  cudaDeviceSynchronize();

  // copy result from GPU to CPU
  cudaMemcpy(A, devA, d*sizeof(float), cudaMemcpyDefault);

  verify_vectorreduction_v1(A_ref, A, d);

  return 0;
}