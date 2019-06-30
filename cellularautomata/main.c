/*
    Cellular Automata 
    Written by: Riccardo Fontanini
    Start date: 5/7/2018
    Note:  
     R O T A S
     O P E R A
     T E N E T
     A R E P O
     S A T O R

*/
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#ifndef Nx
    #define Nx 20
#endif 

#ifndef Ny
    #define Ny 20
#endif

#ifndef MAXITER
    #define MAXITER 50
#endif

#ifndef SIMTIME
  #define SIMTIME 300000
#endif



void calc_neighbours(int field[Nx][Ny], int neighbours[Nx][Ny]) {

    for (int y = 1; y<Ny-1; y++) {
        for (int x = 1; x<Nx-1; x++) {
            neighbours[y][x] = field[y-1][x-1] + field[y-1][x] + field[y-1][x+1] + field[y][x-1] + field[y][x+1] + field[y+1][x-1] + field[y+1][x] + field[y+1][x+1];
        }
    }

}

int apply_rules(int field[Nx][Ny], int neighbours[Nx][Ny], int x, int y) {


    if( neighbours[y][x] == 3 )
        return 1;
    else if( neighbours[y][x] == 2 && field[y][x] == 1)
        return 1;

    return 0;
}

void show (int field[Nx][Ny]) {
    
    for (int y = 0; y<Ny; y++) {
        for (int x = 0; x<Nx; x++) {
            if(field[y][x] == 0)
                fprintf(stderr, "  ");
            else
                fprintf(stderr, "1 ");
        }
        fprintf(stderr, "\n");
    }
}


int main(int argv, char **argc)
{
    int field[Ny][Nx];
    int neighbours[Ny][Nx];

    for (int y = 0; y<Ny; y++) 
        for (int x = 0; x<Nx; x++) {
            field[y][x] = 0;
            neighbours[y][x] = 0;
        }
            
        
    
    #ifdef STOPPER
        field[3][3] = 1;
        field[3][4] = 1;
        field[4][3] = 1;
        field[5][3] = 1;//clear this to lock
        field[5][4] = 1;
        field[5][5] = 1;
        field[5][6] = 1;
        field[6][6] = 1;
    #endif

   

    //glider
    field[4][3] = 1;
    field[5][4] = 1;
    field[5][5] = 1;
    field[4][5] = 1;
    field[3][5] = 1;


    //evolution
    for (int i = 1; i<MAXITER; i++) {
        
        fprintf(stderr, "\033[H\033[J");
        show(field);
        calc_neighbours(field, neighbours);
        //fprintf(stderr, "\n\n\n");
        //show(neighbours);
        for (int y = 1; y<Ny-1; y++) {
            for (int x = 1; x<Nx-1; x++) {
                field[y][x]=apply_rules(field, neighbours, x, y);
            }
        }
        usleep(SIMTIME);
    }

    return 0;
}
