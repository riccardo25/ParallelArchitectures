/*
    Mandelbrot application 
    Written by: Riccardo Fontanini
    Start date: 21 March 2018
    Note:  This program is created to generate mandelbrot things
        Working with complex: https://stackoverflow.com/questions/6418807/how-to-work-with-complex-numbers-in-c

     R O T A S
     O P E R A
     T E N E T
     A R E P O
     S A T O R

*/
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include "log.h"

//debugging 
#define DEBUG

//defined to visualize in console raw data of mandelplot
//#define VISUALIZEMANDELBROTRAW

//max number of iterations to be sure about the point
//#define MAXITERATION 100

#ifndef MANDELBROTLIB_H
#define MANDELBROTLIB_H

typedef struct {
    int rangeH;
    int rangeV;
    int resolution;
    int ** matrix;
    int maxiteration;
    int V_start;
    int V_stop;

}Parallel_params;


/*
Create mandelbrot matrix 
resolution: number of points of reasolution
maxiteration: number of maximum iterations
*/
void createMandelbrot(int ** matrix, int resolution, int maxiteration);

/*
Parallel mandelbrot implementation of createMandelbrot
*/
void createMandelbrot_parallel(int ** matrix, int resolution, int maxiteration);

/*
Function for threading 
*/
void *parallel_thread(void * args);

/**
 * Get horizontal number of pixel, starting from resolution
 * return 0 if resolution is 0
 */
int getHorizontal(int resolution);

/**
 * Get Vertical number of pixel, starting from resolution
 * return 0 if resolution is 0
 */
int getVertical(int resolution);


/**
 * Write on file the mandelbrot picture
 * namefile name of the file to write in
 * Return < 0 when fail
 */
int writePPMMandelbrot(int ** matrix, int resolution, const char* namefile, int maxiteration);


#endif