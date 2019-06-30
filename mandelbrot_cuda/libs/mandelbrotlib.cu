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



__global__ void cuda_elaboration (int *matrix, int maxiteration, int rangeH, int rangeV) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    double c_real = (double)(3 * x) / (double)rangeH - 2;
    double c_img = -1 * (double)(2 * y) / (double)rangeV + 1;
    double zn_real = 0, zn_img = 0, zn1_real = 0, zn1_img = 0;
    double modulesqrd = 0;
    int value = 0;

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
    __syncthreads();

    matrix[y * rangeH + x] = value;
}

__host__ int getHorizontal(int resolution) {
    if (resolution < 0)
        return resolution * 3 * -1;

    return resolution * 3;
}

__host__ int getVertical(int resolution) {
    if (resolution < 0)
        return resolution * 2 * -1;

    return resolution * 2;
}

__host__ int writePPMMandelbrot(int **matrix, int resolution, const char *namefile, int maxiteration) {
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
