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
#include "libs/log.h"


void parameterController(int argv, char **argc, char **namefile, int * resolution, int * maxiteration);
void printHelp();

/*                  MAIN                    */
int main(int argv, char **argc)
{
    char *namefileoutput = NULL;
    int resolution, maxiteration;

    //TIME EVALUATION
    struct timespec start, execution, save;

    //matrix for raw data
    int *matrix;
    FILE *logfile;

    //get current date to create the logfile of this session
    char logname[100];
    time_t now = time(NULL);
    struct tm *t = localtime(&now);
    strftime(logname, sizeof(logname)-1, "../../log/%d-%m-%Y_%H-%M-%S.log", t);
    if ((logfile = fopen (logname, "wb")) == NULL)
    {
        log_error("ERROR in log file creation");
        exit(EXIT_FAILURE);
    }
    log_set_fp(logfile);
    parameterController(argv, argc, &namefileoutput, &resolution, &maxiteration);

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    //creating row data for mandelbrot
    createMandelbrot_parallel(&matrix, resolution, maxiteration);
    //set the clock end creation matrix
    clock_gettime(CLOCK_MONOTONIC_RAW, &execution);

    #ifdef DEBUG
    log_info("Execution time for creation and elaboration: %lu", (execution.tv_sec - start.tv_sec) * 1000000 + (execution.tv_nsec - start.tv_nsec) / 1000);
    #endif

    //write data on disk
    writePPMMandelbrot(&matrix, resolution, namefileoutput, maxiteration);
    clock_gettime(CLOCK_MONOTONIC_RAW, &save);
    #ifdef DEBUG
    log_info("Saving time file: %lu", (save.tv_sec - execution.tv_sec) * 1000000 + (save.tv_nsec - execution.tv_nsec) / 1000);
    #endif

    //free the connection matrix
    free(matrix);
    free(namefileoutput);
    fclose(logfile);

    return 0;
}

void parameterController(int argv, char **argc, char **namefile, int * resolution, int *maxiteration)
{
    *resolution = -1;
    *maxiteration = -1;

    for (int x = 0; x < argv; x++)
    {
        if (argc[x][0] == '-')
        {
            if ( argc[x][1] == 'n')
            {

                *namefile = malloc(strlen(argc[x + 1]) + 7);
                sprintf((*namefile), "./%s.ppm", argc[x + 1]); //next argument is the name
                log_debug("grand %d\n", strlen( (*namefile)));
                #ifdef DEBUG
                log_debug("Name of the file: %s", (*namefile));
                #endif
            }
            else if ( argc[x][1] == 'r')
            {
                *resolution = atoi(argc[x + 1]);
                #ifdef DEBUG
                log_debug("Resolution: %d", *resolution);
                #endif
            }
            else if ( argc[x][1] == 'i')
            {
                *maxiteration = atoi(argc[x + 1]);
                #ifdef DEBUG
                log_debug("Max Iteration: %d", *maxiteration);
                #endif
            }
            else if ( argc[x][1] == 'h')
            {
                printHelp();
                exit(EXIT_SUCCESS);
            }
            else
            {
                printf("Not valid option. Press -h to print help!\n");
                log_error("Error on input parameter");
                exit(EXIT_FAILURE);
            }
        }
    }

    //standard values
    if ( (*namefile) == NULL)
    {
        *namefile = malloc(strlen("./test.ppm") + 1);
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

void printHelp()
{
    printf("\n\nMandelbrot by Riccardo Fontanini\n");
    printf("\n--------------------------------\n");
    printf("\tOPTIONS\n");
    printf("-h \t\tShow help\n");
    printf("-n <name>\t\tchange name of the file, if not specified is test.ppm\n");
    printf("-r <resolution>\tchange the resolution, standard 400\n");
    printf("-i <maxiteration>\tchange the maximum iteration, standard 100\n");
    printf("\n--------------------------------\n\n");
}