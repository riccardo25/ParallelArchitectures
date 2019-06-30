/*
    Mandelbrot application 
    Written by: Riccardo Fontanini
    Start date: 21 March 2018
    Note:  This program is created to generate mandelbrot things
        Working with complex: https://stackoverflow.com/questions/6418807/how-to-work-with-complex-numbers-in-c
        Logging library: https://github.com/rxi/log.c
     R O T A S
     O P E R A
     T E N E T
     A R E P O
     S A T O R

*/
#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <sys/time.h>
#include "libs/mandelbrotlib.h"


__host__ void parameterController(int argv, char **argc, char **namefile, int * resolution, int * maxiteration);
__host__ void printHelp();

/*                  MAIN                    */
int main(int argv, char **argc)
{
    char *namefileoutput = NULL;
    int resolution, maxiteration;
    //TIME EVALUATION
    struct timespec start, execution, save;
    //matrix for raw data
    int *matrix;

    parameterController(argv, argc, &namefileoutput, &resolution, &maxiteration);
    //approsimation of resolution to fit better 
    resolution = (resolution/100)*100;

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    
    //rangeH and rangeV are multiple of 100x100
    const int rangeH = getHorizontal(resolution), rangeV = getVertical(resolution);
    cudaMallocManaged (&matrix, rangeH * rangeV * sizeof (int));
    cudaDeviceSynchronize();

    //blocks 100x100 on grids
    dim3 blocksPerGrid (rangeH/100, rangeV/10);
    dim3 threadsPerBlock (100, 10);
    
    cuda_elaboration <<< blocksPerGrid, threadsPerBlock>>>(matrix, maxiteration, rangeH, rangeV);
    cudaDeviceSynchronize();

    clock_gettime(CLOCK_MONOTONIC_RAW, &execution);
    fprintf(stderr, "\n\nExecution time for creation and elaboration: %lu\n\n", (execution.tv_sec - start.tv_sec) * 1000000 + (execution.tv_nsec - start.tv_nsec) / 1000);
    
    writePPMMandelbrot(&matrix, resolution, namefileoutput, maxiteration);
    clock_gettime(CLOCK_MONOTONIC_RAW, &save);
    fprintf(stderr,"\n\nSaving time file: %lu\n\n", (save.tv_sec - execution.tv_sec) * 1000000 + (save.tv_nsec - execution.tv_nsec) / 1000);
    
    free(namefileoutput);
    return 0;
}

__host__ void parameterController(int argv, char **argc, char **namefile, int * resolution, int *maxiteration) {
    *resolution = -1;
    *maxiteration = -1;

    for (int x = 0; x < argv; x++)
    {
        if (argc[x][0] == '-')
        {

            if ( argc[x][1] == 'n')
            {

                *namefile = (char *) malloc(strlen(argc[x + 1]) + 7);
                sprintf((*namefile), "./%s.ppm", argc[x + 1]); //next argument is the name
                fprintf(stderr,"grand %d\n", strlen( (*namefile)));
                #ifdef DEBUG
                fprintf(stderr,"Name of the file: %s\n", (*namefile));
                #endif
            }
            else if ( argc[x][1] == 'r')
            {
                *resolution = atoi(argc[x + 1]);
                #ifdef DEBUG
                fprintf(stderr,"Resolution: %d\n", *resolution);
                #endif
            }
            else if ( argc[x][1] == 'i')
            {
                *maxiteration = atoi(argc[x + 1]);
                #ifdef DEBUG
                fprintf(stderr,"Max Iteration: %d\n", *maxiteration);
                #endif
            }
            else if ( argc[x][1] == 'h')
            {
                printHelp();
                exit(EXIT_SUCCESS);
            }
            else
            {
                fprintf(stderr, "Not valid option. Press -h to print help!\n");
                fprintf(stderr, "Error on input parameter\n");
                exit(EXIT_FAILURE);
            }
        }
    }

    //standard values
    if ( (*namefile) == NULL)
    {
        *namefile = (char *) malloc(strlen("./test.ppm") + 1);
        strcpy((*namefile), "./test.ppm");
    }

    if( *resolution <= 0)
    {
        *resolution = 400;
    }

    if( *maxiteration <= 0)
    {
        *maxiteration = 100;
    }
}

__host__ void printHelp() {
    printf("\n\nMandelbrot by Riccardo Fontanini\n");
    printf("\n--------------------------------\n");
    printf("\tOPTIONS\n");
    printf("-h \t\tShow help\n");
    printf("-n <name>\t\tchange name of the file, if not specified is test.ppm\n");
    printf("-r <resolution>\tchange the resolution, standard 400\n");
    printf("-i <maxiteration>\tchange the maximum iteration, standard 100\n");
    printf("\n--------------------------------\n\n");
}