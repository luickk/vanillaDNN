#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef void (*activationFunc)(double inp, double out);

typedef struct {
  double *bias;
  double *weights;
  activationFunc actFunc;
} baseLayer;

/*
general NN functions
*/
void initLayer(double size, baseLayer &layer, activationFunc actFunc) {
  layer.bias = (double*)malloc(size * sizeof(double));
  layer.weights = (double*)malloc(size * sizeof(double));
  layer.actFunc = actFunc;
}

/*
Layer Operations
*/

void testActFunc(double inp, double out){

}

int main(){
  printf("test \n");
  baseLayer hiddenLayer1;
  initLayer(100, hiddenLayer1, testActFunc);
}
