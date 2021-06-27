#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef void (*activationFunc)(float inp, float out);

enum layerType {
    fullyConnected,
};

typedef struct {
  float *bias;
  float *weights;
  activationFunc actFunc;
  int size;
  layerType type;
} baseLayer;

/*
util functions
*/
int trainDNN(char pathToFile[]) {
  FILE *fp;
  char *line = 0;
  float inp[4] = {0};
  int inpIt = 0;
  size_t len = 0;
  ssize_t read;

  fp = fopen(pathToFile, "r");
  if (fp == 0) {
   return 1;
  }
  while ((read = getline(&line, &len, fp)) != -1) {
    if (inpIt == 4) {
      printf("\n ");
      for(int j = 0; j <4; j++) {
        printf("%f ", inp[j]);
      }
      inpIt = 0;
    }
    inp[inpIt] = atof(line);
    inpIt++;
  }
  fclose(fp);
  if (line) {
   free(line);
  }
  return 0;
}

/*
general NN functions
*/
void initLayer(float size, layerType type, baseLayer &layer, activationFunc actFunc) {
  layer.bias = (float*)malloc(size * sizeof(float));
  layer.weights = (float*)malloc(size * sizeof(float));
  layer.actFunc = actFunc;
  layer.size = size;
  layer.type = type;
}

baseLayer *createLayer(float size, layerType type, activationFunc actFunc) {
  static baseLayer layer;
  initLayer(size, fullyConnected, layer, actFunc);
  return &layer;
}

/*
Layer Operations
*/
void testActFunc(float inp, float out){
  if (inp < 1) {
    out = 0;
  } else if (inp >=1) {
    out = 1;
  }
}

int main(){
  int nPredict = 5;

  printf("nPredict: %d \n", nPredict);

  baseLayer *inpLayer = createLayer(nPredict, fullyConnected, testActFunc);
  baseLayer *hiddenLayer2 = createLayer(100, fullyConnected, testActFunc);
  baseLayer *hiddenLayer1 = createLayer(100, fullyConnected, testActFunc);
  baseLayer *outpLayer = createLayer(nPredict, fullyConnected, testActFunc);

  int rc = trainDNN("../data/datasetByLine.csv");
  if (rc != 0) {
     printf("file open failed");
     return 0;
  }
}
