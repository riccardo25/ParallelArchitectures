/*
    PDE solver
    Written by: Riccardo Fontanini
    Start date: 7 May 2018
    Note:  This program is a PDE solver
     R O T A S
     O P E R A
     T E N E T
     A R E P O
     S A T O R

*/
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <sys/time.h>

#ifndef N_X
  #define N_x 10
#endif
#ifndef N_y
  #define N_y 10
#endif
#ifndef BLOCKDIM
  #define BLOCKDIM 32
#endif
#ifndef MAXITER
  #define MAXITER 50
#endif

#ifndef SIMTIME
  #define SIMTIME 200000
#endif



__global__ void solve(double *grid, int odd) {
  //odd means: in odd rows swap odd, x%2 == odd -> elaborate
  
  int x = blockIdx.x * BLOCKDIM + threadIdx.x;
  int y = blockIdx.y * BLOCKDIM + threadIdx.y;
  
  if(y % 2 == 1 && odd == 0)
    odd = 1;
  else if(y % 2 == 1 && odd == 1)
    odd = 0;

  if(y == 0 || y >= N_y-1 || x == 0 || x >= N_x-1 || x % 2 == odd)
    return;
  
  *(grid + y*N_x + x) = 0.2 * (*(grid + y*N_x + x) + *(grid + (y-1)*N_x + x) + *(grid + (y+1)*N_x + x) + *(grid + y*N_x + x - 1) + *(grid + y*N_x + x + 1));
}

__host__ void solve_CPU(double *grid){

  int odd =0;
  for (int i = 0; i<MAXITER; i++) {
    odd = i%2;
    for (int y = 0; y < N_y; y++) {
      for (int x = 0; x < N_x; x++) {
        if(y == 0 || y >= N_y-1 || x == 0 || x >= N_x-1 || x % 2 == odd)
          continue;
        *(grid + y*N_x + x) = 0.2 * (*(grid + y*N_x + x) + *(grid + (y-1)*N_x + x) + *(grid + (y+1)*N_x + x) + *(grid + y*N_x + x - 1) + *(grid + y*N_x + x + 1));
      }
    }

    #ifdef SIMULATION
      fprintf(stderr,"\033c");
      fprintf (stderr, "Ciclo: %d\n", i);
      fprintf (stderr, "\tGRID \n\n");
      double *ptrP = grid;
      for (int y = 0; y < N_y; y++) {
        for (int x = 0; x < N_x; x++, ptrP++)
          fprintf (stderr, "%3.f ", *ptrP);
        fprintf (stderr, "\n");
      }
      usleep(SIMTIME);
    #endif


  }
  
}



__global__ void set_default(double *grid, double def) {
  
  int x = blockIdx.x * BLOCKDIM + threadIdx.x;
  int y = blockIdx.y * BLOCKDIM + threadIdx.y;

  if(y == 0 || y >= N_y-1 || x == 0 || x >= N_x-1)
    return;
  
  *(grid + y*N_x + x) = def;
}


int main () {
  
  double *grid;
  int gridx                 = N_x / BLOCKDIM;
  int gridy                 = N_y / BLOCKDIM;
  if( N_x % BLOCKDIM != 0)
    gridx++;
  if(N_y % BLOCKDIM != 0)
    gridy++;
  double toll = 0; //tollerance checked every cycle ifdef TOLLERANCE
  //TIME EVALUATION
  struct timespec start, execution;

  fprintf (stderr, "grid x : %d y: %d \n\n", gridx, gridy);
  dim3 blocksPerGrid (gridx, gridy);
  dim3 threadsPerBlock (BLOCKDIM, BLOCKDIM);
  cudaMallocManaged ( &(grid), N_x * N_x * sizeof (double) );

  /* SET BOUNDARY */
  grid[0] = 120;
  grid[1] = 120;
  grid[2] = 120;
  grid[3] = 120;
  /* SET DEFAULT VALUE TO SPACE */
  cudaDeviceSynchronize ();
  set_default <<<blocksPerGrid, threadsPerBlock>>> (grid, 50);
  cudaDeviceSynchronize ();

  #ifdef ELABCPU
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    solve_CPU(grid);
    clock_gettime(CLOCK_MONOTONIC_RAW, &execution);
    fprintf(stderr, "\n\nExecution time for CPU: %lu\n\n", (execution.tv_sec - start.tv_sec) * 1000000 + (execution.tv_nsec - start.tv_nsec) / 1000);
  
  
    #ifdef SHOWRESULT
    fprintf (stderr, "\tGRID CPU\n\n");
    double *ptrC = grid;
    for (int y = 0; y < 10; y++) {
      for (int x = 0; x < 10; x++, ptrC++)
        fprintf (stderr, "%3.f ", *ptrC);
      ptrC = grid + N_x * (y+1);
      fprintf (stderr, "\n");
    }
    #endif
  #endif

  cudaDeviceSynchronize ();
  set_default <<<blocksPerGrid, threadsPerBlock>>> (grid, 50);
  cudaDeviceSynchronize ();

  for (int i =0; i<MAXITER; i++){
    
    solve <<<blocksPerGrid, threadsPerBlock>>> (grid, i%2);
    cudaDeviceSynchronize ();
    
    #ifdef SIMULATION
      
      fprintf(stderr,"\033c");
      fprintf (stderr, "Ciclo: %d\n", i);
      fprintf (stderr, "\tGRID \n\n");
      double *ptrP = grid;
      for (int y = 0; y < N_y; y++) {
        for (int x = 0; x < N_x; x++, ptrP++)
          fprintf (stderr, "%3.f ", *ptrP);
        fprintf (stderr, "\n");
      }
      usleep(SIMTIME);
    #endif

    #ifdef TOLLERANCE
      double newtoll = 0; 
      double *point = grid;
      for (int y = 0; y < N_y; y++) {
        for (int x = 0; x < N_x; x++, point++)
          newtoll += abs(*point);
      }

      if( abs(newtoll - toll) < TOLLERANCE){
        fprintf(stderr, "Breaked for tollerance trigger!\n");
        break;
      }
      else
        toll = newtoll;

    #endif
  }
  
  #ifdef SHOWRESULT
    fprintf (stderr, "\tGRID GPU\n\n");
    double *ptrG = grid;
    for (int y = 0; y < 10; y++) {
      for (int x = 0; x < 10; x++, ptrG++)
        fprintf (stderr, "%3.f ", *ptrG);
      ptrG = grid + N_x * (y+1);
      fprintf (stderr, "\n");
    }
  #endif

  return 0;

}
