#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "utils.cuh"

int main(void) {

  // pointer for Host memory
  //float *A;
  //float *A_ref;

  float *B;
  float *B_ref;

  // pointer for Device memory
  //float *devA;
  float *devB;

  const int N = 256;
  const int d = 128;

  // allocate host memory
  //cudaMallocHost(&A, d*sizeof(float));
  //cudaMallocHost(&A_ref, d*sizeof(float));

  cudaMallocHost(&B, N*d*sizeof(float));
  cudaMallocHost(&B_ref, N*d*sizeof(float));

  // initialize vector/matrix
  //initArray(A, d);
  initMatrix(B, N, d);

  // initialize referance vector/matrix
  //copyArray(A, A_ref, d);
  copyMatrix(B, B_ref, N, d);
  
  // allocate memory on Device
  //cudaMalloc(&devA, d*sizeof(float));
  cudaMalloc(&devB, N*d*sizeof(float));

  // copy from CPU to GPU
  //cudaMemcpy(devA, A, d*sizeof(float), cudaMemcpyDefault);
  cudaMemcpy(devB, B, N*d*sizeof(float), cudaMemcpyDefault);

  test_matrixReduction_v1(B, N, d);

  // wait for kernel to be done
  cudaDeviceSynchronize();

  // copy result from GPU to CPU
  //cudaMemcpy(A, devA, d*sizeof(float), cudaMemcpyDefault);
  cudaMemcpy(B, devB, N*d*sizeof(float), cudaMemcpyDefault);

  //verify_vectorReduction_v1(A_ref, A, d);
  verify_matrixReduction_v1(B_ref, B, N, d);

  return 0;
}