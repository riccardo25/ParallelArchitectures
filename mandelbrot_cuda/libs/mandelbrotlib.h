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

//debugging 
#define DEBUG

//defined to visualize in console raw data of mandelplot
//#define VISUALIZEMANDELBROTRAW

//max number of iterations to be sure about the point
//#define MAXITERATION 100

#ifndef MANDELBROTLIB_H
#define MANDELBROTLIB_H



/**
 * KERNEL of cuda elaboration
 */
__global__ void cuda_elaboration (int *matrix, int maxiteration, int rangeH, int rangeV);

/**
 * Get horizontal number of pixel, starting from resolution
 * return 0 if resolution is 0
 */
__host__ int getHorizontal(int resolution);

/**
 * Get Vertical number of pixel, starting from resolution
 * return 0 if resolution is 0
 */
__host__ int getVertical(int resolution);


/**
 * Write on file the mandelbrot picture
 * namefile name of the file to write in
 * Return < 0 when fail
 */
__host__ int writePPMMandelbrot(int ** matrix, int resolution, const char* namefile, int maxiteration);


#endif