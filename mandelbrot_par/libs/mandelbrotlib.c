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

#include "mandelbrotlib.h"

void createMandelbrot(int **matrix, int resolution, int maxiteration)
{
    const int rangeH = getHorizontal(resolution), rangeV = getVertical(resolution);
    *matrix = malloc(rangeV * rangeH * sizeof(int)); //using heap insted stack! -> 'cose limitation in memory
    double zn_real = 0, zn_img = 0, zn1_real = 0, zn1_img = 0;
    double modulesqrd = 0;
    double c_real, c_img;
    int value = 0;
    for ( int y = 0; y < rangeV; y++ ) { //iterate all points of the "image"
        for ( int x = 0; x < rangeH; x++ ) {
            c_real = (double)(3 * x) / (double)rangeH - 2;
            c_img = -1 * (double)(2 * y) / (double)rangeV + 1;
            zn_real = 0;
            zn_img = 0;
            value = 0;
            zn1_img = 0;
            zn1_real = 0;


            for (int a = 0; a < maxiteration; a++) {
                zn1_real = c_real + zn_real * zn_real - zn_img * zn_img;
                zn1_img = c_img + 2 * zn_real * zn_img;
                modulesqrd = zn1_real * zn1_real + zn1_img * zn1_img;
                if (modulesqrd > 4.0) {
                    value = a + 1; 
                    break;
                }
                zn_real = zn1_real;
                zn_img = zn1_img;
            }

            *( *( matrix )+( y*rangeH+x )) = value;
        }
    }
}

void createMandelbrot_parallel(int ** matrix, int resolution, int maxiteration)
{
    const int rangeH = getHorizontal(resolution), rangeV = getVertical(resolution);
    *matrix = malloc(rangeV * rangeH * sizeof(int)); //using heap insted stack! -> 'cose limitation in memory
    
    int number_of_threads = 8;
    #ifdef NUMBER8THREADS
        number_of_threads = 8;
    #endif
    pthread_t threads[number_of_threads];
    Parallel_params params[number_of_threads];

    for (int i = 0; i<number_of_threads; i++) {
        //PARAMETERS CLUSTERING
        params[i].rangeH = rangeH;
        params[i].rangeV = rangeV;
        params[i].resolution = resolution;
        params[i].matrix = matrix;
        params[i].maxiteration = maxiteration;
        //choose vertical segmentation 
        params[i].V_start = (rangeV/number_of_threads) * i;
        int stop = (rangeV/number_of_threads) * (i+1);
        if(stop > rangeV)
            stop = rangeV;
        params[i].V_stop = stop;
        pthread_create(&threads[i], NULL, parallel_thread, (void *) &params[i]);
    }

    for(int j=0; j < number_of_threads; j++) {
        pthread_join( threads[j], NULL);
    }

}

void *parallel_thread(void * args) {

    Parallel_params *params = (Parallel_params *) args;
    const int rangeH = params->rangeH, rangeV = params->rangeV;
    double zn_real = 0, zn_img = 0, zn1_real = 0, zn1_img = 0;
    double modulesqrd = 0;
    double c_real, c_img;
    int value = 0;

    for ( int y = params->V_start; y < params->V_stop; y++ ) {        
        for ( int x = 0; x < rangeH; x++ ) {
            c_real = (double)(3 * x) / (double)rangeH - 2;
            c_img = -1 * (double)(2 * y) / (double)rangeV + 1;
            zn_real = 0;
            zn_img = 0;
            value = 0;
            zn1_img = 0;
            zn1_real = 0;
            for (int a = 0; a < params->maxiteration; a++) {
                zn1_real = c_real + zn_real * zn_real - zn_img * zn_img;
                zn1_img = c_img + 2 * zn_real * zn_img;
                modulesqrd = zn1_real * zn1_real + zn1_img * zn1_img;
                if (modulesqrd > 4.0) {
                    value = a + 1; 
                    break;
                }
                zn_real = zn1_real;
                zn_img = zn1_img;
            }
            (*(params->matrix))[y * rangeH + x] = value;
        }
    }
}


int getHorizontal(int resolution) {
    if (resolution < 0)
        return resolution * 3 * -1;

    return resolution * 3;
}

int getVertical(int resolution) {
    if (resolution < 0)
        return resolution * 2 * -1;

    return resolution * 2;
}

int writePPMMandelbrot(int **matrix, int resolution, const char *namefile, int maxiteration) {
    FILE *file;
    const int rangeH = getHorizontal(resolution), rangeV = getVertical(resolution);
    srand(time(NULL));
    if ((file = fopen(namefile, "wb")) == NULL) {
        return -1;
    }
    fprintf(file, "P6\n%d %d\n255\n", rangeH, rangeV);
   
   //color palette
    unsigned char palette[maxiteration + 1][3]; //values go from 0 to MAXITERATIONS (Evry number has different color)
    for (int i = 0; i < maxiteration + 1; i++) {
        palette[i][0] = (unsigned char)(rand() % 256);
        palette[i][1] = (unsigned char)(rand() % 256);
        palette[i][2] = (unsigned char)(rand() % 256);
    }

    for (int y = 0; y < rangeV; y++) {
        for (int x = 0; x < rangeH; x++) {
            fwrite(palette[(*matrix)[y * rangeH + x]], 1, 3, file);
        }
    }
    fclose(file);
    return 1;
}
