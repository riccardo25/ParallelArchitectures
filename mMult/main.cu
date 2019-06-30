/*
    Mmult application 
    Written by: Riccardo Fontanini
    Start date: 3 May 2018
    Note:  This program is created to multiply 3 matrix
     R O T A S
     O P E R A
     T E N E T
     A R E P O
     S A T O R

*/
#include <stdio.h>

#ifndef N
  #define N 1024
#endif

#ifndef BLOCKDIM
  #define BLOCKDIM 32
#endif


void __global__ shared_mmult(int *A, int *B, int *C) {
  __shared__ int s_A [BLOCKDIM * BLOCKDIM];
  __shared__ int s_B [BLOCKDIM * BLOCKDIM];
  int value = 0;
  int block_index = threadIdx.y * BLOCKDIM + threadIdx.x;
  int xa, ya, xb, yb = 0;
  ya = blockIdx.y;
  xb = blockIdx.x;
  /*
  se matrice 3*3 e multipla di 32
  (blockIdx.y; blockIdx.c) == (0, 0) -> siamo nel blocco C00 = A00 * B00 + A01 * B10 + A02 * B20
  (blockIdx.y; blockIdx.x) == (0, 1) -> siamo nel blocco C01 = A00 * B01 + A01 * B11 + A02 * B21
  (blockIdx.y; blockIdx.x) == (1, 0) -> siamo nel blocco C10 = A10 * B00 + A11 * B10 + A12 * B20
  (blockIdx.y; blockIdx.x) == (1, 1) -> siamo nel blocco C11 = A10 * B01 + A11 * B11 + A12 * B21
  e così via...

  */

  for(int i = 0; i < N/BLOCKDIM; i++) {
    //copy global memory Axaya and Bxbyb to shared memory
    xa = i;
    yb = i;
    *(s_A + block_index) = *(A + ( (ya * BLOCKDIM) + threadIdx.y ) * N + (threadIdx.x + xa * BLOCKDIM));
    *(s_B + block_index) = *(B + ( (yb * BLOCKDIM) + threadIdx.y ) * N + (threadIdx.x + xb * BLOCKDIM));

    if(blockIdx.y * BLOCKDIM + threadIdx.y >= N || blockIdx.x * BLOCKDIM + threadIdx.x >= N)
      return;

    __syncthreads();

    for (int k = 0; k < BLOCKDIM; k++)
      value += *(s_A + threadIdx.y * BLOCKDIM + k) * *(s_B + k * BLOCKDIM + threadIdx.x);

    __syncthreads();
  }

  *(C + ( (blockIdx.y * BLOCKDIM) + threadIdx.y ) * N + (threadIdx.x + blockIdx.x * BLOCKDIM)) = value;

}

void __global__ simple_mmult(int *A, int *B, int *C) {
  int value = 0;
  for (int k = 0; k < N; k++)
    value += *(A + ( (blockIdx.y * BLOCKDIM) + threadIdx.y ) * N + (k + blockIdx.x * BLOCKDIM) ) * *(B + ( (blockIdx.y * BLOCKDIM) + k ) * N + (threadIdx.x + blockIdx.x * BLOCKDIM));
  *(C + ( (blockIdx.y * BLOCKDIM) + threadIdx.y ) * N + (threadIdx.x + blockIdx.x * BLOCKDIM)) =  value;

}


int main () {

  int NN = N * N;
  int *A, *B, *C;
  cudaMallocManaged (&A, NN * sizeof (int));
  cudaMallocManaged (&B, NN * sizeof (int));
  cudaMallocManaged (&C, NN * sizeof (int));
  cudaDeviceSynchronize();
  fprintf(stderr, "PRIma\n");
  int *ptrA = A;
  int *ptrB = B;
  int *ptrC = C;
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++, ptrA++, ptrB++, ptrC++) {
      *ptrA = i * N + j;
      *ptrB = *ptrC = 0;
      if (i == j) *ptrB = 1;
    }
  cudaDeviceSynchronize();
  fprintf(stderr, "seconda\n");

  fprintf (stderr, "La visualizzazione delle matrici è stata limitata a 10x10\n");
  fprintf (stderr, "Matrix A\n");
  ptrA = A;
  for (int i = 0; i < 10; i++) {
    ptrA = A + i * N;
    for (int j = 0; j < 10; j++, ptrA++)
      fprintf (stderr, "%2d ", *ptrA);
    fprintf (stderr, "\n");
  }
  fprintf (stderr, "\n\n\n");

  fprintf (stderr, "Matrix B\n");
  ptrB = B;
  for (int i = 0; i < 10; i++) {
    ptrB = B + i * N;
    for (int j = 0; j < 10; j++, ptrB++)
      fprintf (stderr, "%2d ", *ptrB);
    fprintf (stderr, "\n");
  }
  fprintf (stderr, "\n\n\n");
  cudaDeviceSynchronize ();
  dim3 blocksPerGrid (N/BLOCKDIM, N/BLOCKDIM);
  dim3 threadsPerBlock (BLOCKDIM, BLOCKDIM);
  #ifdef SIMPLE
    simple_mmult <<< blocksPerGrid, threadsPerBlock>>> (A, B, C);
    cudaDeviceSynchronize ();
    ptrC = C;
    fprintf (stderr, "Matrix C SIMPLE (not shared memory)\n");
    for (int i = 0; i < 10; i++) {
      ptrC = C + i * N;
      for (int j = 0; j < 10; j++, ptrC++)
        fprintf (stderr, "%2d ", *ptrC);
      fprintf (stderr, "\n");
    }
    fprintf (stderr, "\n\n\n");
  #endif
  int dimSHARED = 3 * ( BLOCKDIM * BLOCKDIM * sizeof(int));
  shared_mmult <<< blocksPerGrid, threadsPerBlock>>> (A, B, C);
  cudaDeviceSynchronize ();

  fprintf (stderr, "Matrix C Shared memory\n");
  ptrC = C;
  for (int i = 0; i < 10; i++) {
    ptrC = C + i * N;
    for (int j = 0; j < 10; j++, ptrC++)
      fprintf (stderr, "%2d ", *ptrC);
    fprintf (stderr, "\n");
  }

  return 0;

}
