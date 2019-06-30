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
#include <math.h>

#ifndef N
  #define N 2048
#endif

#ifndef BLOCKDIM
  #define BLOCKDIM 32
#endif

#ifndef NROWSTESLA
  #define NROWSTESLA 2048
#endif

//start and end rows define which submatrix multiply
void __global__ shared_mmult_tesla(int *A, int *B, int *C, int startrow, int endrow) {
  __shared__ int s_A [BLOCKDIM * BLOCKDIM];
  __shared__ int s_B [BLOCKDIM * BLOCKDIM];
  int value = 0;
  int block_index = threadIdx.y * BLOCKDIM + threadIdx.x;
  int xa, ya, xb, yb = 0;
  ya = (startrow / BLOCKDIM) + blockIdx.y;
  xb = blockIdx.x;
  //check boundary
  if(startrow + blockIdx.y * BLOCKDIM + threadIdx.y >= N || startrow + blockIdx.y * BLOCKDIM + threadIdx.y >= endrow || blockIdx.x * BLOCKDIM + threadIdx.x >= N)
    return;
    
  for(int i = 0; i < N/BLOCKDIM; i++) {
    //copy global memory Axaya and Bxbyb to shared memory
    xa = i;
    yb = i;
    *(s_A + block_index) = *(A + ( (ya * BLOCKDIM) + threadIdx.y ) * N + (threadIdx.x + xa * BLOCKDIM));
    *(s_B + block_index) = *(B + ( (yb * BLOCKDIM) + threadIdx.y ) * N + (threadIdx.x + xb * BLOCKDIM));

    __syncthreads();

    for (int k = 0; k < BLOCKDIM; k++)
      value += *(s_A + threadIdx.y * BLOCKDIM + k) * *(s_B + k * BLOCKDIM + threadIdx.x);

    __syncthreads();
  }

  *(C + ( startrow + (blockIdx.y * BLOCKDIM) + threadIdx.y ) * N + (threadIdx.x + blockIdx.x * BLOCKDIM)) = value;
}


void __global__ shared_mmult_gtx(int *A, int *B, int *C, int startrow, int endrow) {
  __shared__ int s_A [BLOCKDIM * BLOCKDIM];
  __shared__ int s_B [BLOCKDIM * BLOCKDIM];
  int value = 0;
  int block_index = threadIdx.y * BLOCKDIM + threadIdx.x;
  int xa, ya, xb, yb = 0;
  ya = (startrow / BLOCKDIM) + blockIdx.y;
  xb = blockIdx.x;
  //check boundary
  if(startrow + blockIdx.y * BLOCKDIM + threadIdx.y >= N || startrow + blockIdx.y * BLOCKDIM + threadIdx.y >= endrow || blockIdx.x * BLOCKDIM + threadIdx.x >= N)
    return;
    
  for(int i = 0; i < N/BLOCKDIM; i++) {
    //copy global memory Axaya and Bxbyb to shared memory
    xa = i;
    yb = i;
    *(s_A + block_index) = *(A + ( (ya * BLOCKDIM) + threadIdx.y ) * N + (threadIdx.x + xa * BLOCKDIM));
    *(s_B + block_index) = *(B + ( (yb * BLOCKDIM) + threadIdx.y ) * N + (threadIdx.x + xb * BLOCKDIM));

    __syncthreads();

    for (int k = 0; k < BLOCKDIM; k++)
      value += *(s_A + threadIdx.y * BLOCKDIM + k) * *(s_B + k * BLOCKDIM + threadIdx.x);

    __syncthreads();
  }

  *(C + ( startrow + (blockIdx.y * BLOCKDIM) + threadIdx.y ) * N + (threadIdx.x + blockIdx.x * BLOCKDIM)) = value;
}


int main () {

  //N: number of elements of matrix
  int NN                              = N * N;
  int *A                              = (int *) malloc(NN*sizeof(int));
  int *B                              = (int *) malloc(NN*sizeof(int));
  int *C                              = (int *) malloc(NN*sizeof(int));
  int griddim                         = N / BLOCKDIM;
  int *ptrA                           = A;
  int *ptrB                           = B;
  int *ptrC;

  if( N % BLOCKDIM != 0) 
    griddim++;

  
  
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  fprintf (stderr, "Number of devices found: %d \n\n", nDevices);
  cudaDeviceSynchronize();

  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++, ptrA++, ptrB++) {
      *ptrA = i * N + j;
      *ptrB = 0;
      if (i == j) *ptrB = 1;
    }

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
  //pointer vector arrays
  int *A_d[nDevices], *B_d[nDevices], *C_d[nDevices];
  
  for (int i = 0; i<nDevices; i++) {
    cudaSetDevice(i);
    //allocate memory both GPU card and GPU
    cudaMalloc (&(A_d[i]), NN * sizeof (int));
    cudaMalloc (&(B_d[i]), NN * sizeof (int));
    cudaMalloc (&(C_d[i]), NN * sizeof (int));
    //copy memory from host to global memory of i-GPU card
    //cudamemcopy is async
    cudaMemcpy (A_d[i], A, NN * sizeof (int), cudaMemcpyHostToDevice);
    cudaMemcpy (B_d[i], B, NN * sizeof (int), cudaMemcpyHostToDevice);
  }

  //sync to be sure device memory is ok
  for (int i = 0; i<nDevices; i++) {
    cudaSetDevice(i);
    cudaDeviceSynchronize();
  }

  //kernel is not blocking, so kernels run parallel
  cudaSetDevice(0);
  dim3 blocksPerGridTESLA (N/BLOCKDIM, NROWSTESLA/BLOCKDIM);
  dim3 threadsPerBlockTESLA (BLOCKDIM, BLOCKDIM);
  shared_mmult_tesla <<< blocksPerGridTESLA, threadsPerBlockTESLA >>> (A_d[0], B_d[0], C_d[0], 0, NROWSTESLA);
  cudaSetDevice(1);
  dim3 blocksPerGridGTX (N/BLOCKDIM, (N - NROWSTESLA)/BLOCKDIM);
  dim3 threadsPerBlockGTX (BLOCKDIM, BLOCKDIM);
  shared_mmult_gtx <<< blocksPerGridGTX, threadsPerBlockGTX >>> (A_d[1], B_d[1], C_d[1], NROWSTESLA, N);

  //join
  for (int i = 0; i<nDevices; i++) {
    cudaSetDevice(i);
    cudaDeviceSynchronize();
  }

  cudaMemcpy (C, C_d[0], NROWSTESLA * N * sizeof (int), cudaMemcpyDeviceToHost);
  cudaMemcpy ( (C + NROWSTESLA * N ), (C_d[1] + NROWSTESLA * N), (NN - NROWSTESLA * N ) * sizeof (int), cudaMemcpyDeviceToHost);

  //check
  fprintf (stderr, "Matrix C\n");
  unsigned long diff = 0;
  ptrC = C;
  ptrA = A;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++, ptrC++, ptrA++) {
      if( j < 10 && i < 10)
        fprintf (stderr, "%2d ", *ptrC);
      diff += abs( *ptrA - *ptrC );

    }
    if( i < 10 )
      fprintf (stderr, "\n");
  }
  fprintf (stderr, "Differenza: %lu\n", diff);
  free(A);
  free(B);
  
  for (int i = 0; i<nDevices; i++) {
    cudaSetDevice(i);
    cudaFree(A_d[i]);
    cudaFree(B_d[i]);
    cudaFree(C_d[i]);
  }

  return 0;

}
