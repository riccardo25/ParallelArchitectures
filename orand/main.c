/*
    Neuron application 
    Written by: Riccardo Fontanini
    Start date: 5 May 2018
    Note:  This program simulates a neuron
     R O T A S
     O P E R A
     T E N E T
     A R E P O
     S A T O R

*/
#include <stdio.h>
#include <stdlib.h>

#ifndef NWEIGHT
  #define NWEIGHT 3
#endif

#ifndef MAXITER
  #define MAXITER 200
#endif

#ifndef LR
  #define LR 0.5
#endif

#ifndef W0
  #define W0 -6
#endif

#ifndef W1
  #define W1 10
#endif

#ifndef W2
  #define W2 1
#endif

double sign(double input)
{
  double out = -1;
  if(input > 0)
    out = 1;
  return out;
}

double neuron (double *x, double *w) {
  double scalprod = 0;

  for (int i = 0; i< NWEIGHT; i++) {
    scalprod += w[i] * x[i];
  }

  return sign(scalprod);

}

int main () {

  double w [ NWEIGHT ]= {W0, W1, W2}; //third value is bias
  double learning_rate = LR;
  //generate training set
  const int train_len = 4;
  double x_in[][NWEIGHT] = { { -1, -1, 1 }, { -1, 1, 1 }, { 1, -1, 1 }, { 1, 1, 1 } };//THIRD IS BIAS
  double out[] = {-1, -1, -1, 1};//AND
  #ifdef OR
    out[1] = 1;
    out[2] = 1;
  #endif
  
  //HEBBIAN LEARNING
  for (int k = 0; k<MAXITER; k++) {
    int index_train = k % train_len;
    fprintf(stderr, "Ciclo %d\n", k);

    for ( int i = 0; i<NWEIGHT; i++ ) {
      w[i] = w[i] + learning_rate * x_in[index_train][i] * out[index_train];
    }

    //test of data
    int test = 0;
    for ( int i = 0; i<train_len; i++ ) {
      if(neuron(x_in[i], w) != out[i] )
        test= 1;
    }
    if(test == 0)
      break;
    
    
  }

  fprintf(stderr, "W: %f %f %f\n", w[0], w[1], w[2]);

  //LETS PREDICT
  double x[NWEIGHT] = {-1, 1, 1};  //THIRD IS BIAS
  double result = neuron(x, w);

  fprintf(stderr, "Result: %2.f\n", result);

  return 1;

}
