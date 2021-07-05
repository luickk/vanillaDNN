#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef void (*activationFunc)(float *inp, float *out);

enum layerType {
    fullyConnected,
};

typedef struct {
  float *bias;
  float *weights;
  float *nodes;
  activationFunc actFunc;
  int size;
  layerType type;
} baseLayer;

typedef struct {
  int nLayer;
  baseLayer *nnLayer[];
} neuralNet;

/*
general NN functions
*/
void initLayer(float size, layerType type, baseLayer &layer, activationFunc actFunc) {
  layer.bias = (float*)malloc(size * sizeof(float));
  layer.weights = (float*)malloc(size * sizeof(float));
  layer.nodes = (float*)malloc(size * sizeof(float));

  layer.actFunc = actFunc;
  layer.size = size;
  layer.type = type;
}

baseLayer *createLayer(float size, layerType type, activationFunc actFunc) {
  static baseLayer layer;
  initLayer(size, fullyConnected, layer, actFunc);
  return &layer;
}

void initWeights(baseLayer *layer, int size){
  int randW = 0;
  for (int i = 0; i < size; i++) {
    randW = (rand() % (10 - 0 + 1)) + 0;
    layer->weights[i] = randW;
  }
}

void initBias(baseLayer *layer, int size){
  int randB = 0;
  for (int i = 0; i < size; i++) {
    randB = (rand() % (2 - 0 + 1)) + 0;
    layer->bias[i] = randB;
  }
}

neuralNet *createNet(baseLayer *layer[], int nLayer) {
  neuralNet *nn = (neuralNet *)malloc(sizeof(baseLayer)*nLayer+sizeof(nLayer));
  for (int i=0; i<nLayer; ++i) {
     nn->nnLayer[i]=layer[i];
  }
  nn->nLayer = nLayer;

  for (int i=0; i<nLayer; ++i) {
    initWeights(nn->nnLayer[i],nn->nnLayer[i]->size);
    initBias(nn->nnLayer[i],nn->nnLayer[i]->size);
  }

  return nn;
}

/*
Layer Operations
*/

// fast Sigmoid
// from https://stackoverflow.com/a/10733861
void sigmoidActFunc(float *x, float *y){
   *y = *x / (1 + abs(*x));
}

void reluActFunc(float *x, float *y)
{
  *y = fmax(0, *x);
}
void reluDerivativeActFunc(float *x, float *y)
{
  if (*x > 0) {
    *y = 1;
  } else if (*x <= 0){
    *y = 0;
  }
}
void noActFunc(float *x, float *y)
{
  *y = *x;
}


// outputs Delta as sensitives to output layer
void calcOutpWeight(neuralNet *net, int nPredict){
  float y = 0;
  for (int i = 0; i < nPredict; i++) {
    reluDerivativeActFunc(&net->nnLayer[net->nLayer]->nodes[i], &y);
    net->nnLayer[net->nLayer]->weights[i] =  net->nnLayer[net->nLayer]->nodes[i] * (y * (net->nnLayer[0]->nodes[i] - net->nnLayer[net->nLayer]->nodes[i]));
  }
}

void calcNodeValForLayer(baseLayer *lastLayer, baseLayer *layer) {
  for(int i = 0; i<layer->size; i++) {
    for (int j = 0; j<lastLayer->size; j++) {
      layer->nodes[i] += layer->weights[i] * lastLayer->nodes[j];
      printf("Neuron: %i, last layer neuron: %i: %.0f \n", i, j, layer->nodes[i]);
    }
    layer->nodes[i] += layer->bias[i];
    layer->actFunc(&layer->nodes[i], &layer->nodes[i]);
  }
}

void backpropagate(neuralNet *net, float input[4], int nPredict) {
  calcOutpWeight(net, nPredict);

  for(int i = net->nLayer-1; i > net->nLayer; i--) {
    // net->nnLayer[i]->weights = net->nnLayer[i+]->weights*net->nnLayer[i+].weights*
  }
}

void feedForward(neuralNet *net, float input[4], int nPredict){
  // setting input to input layer nodes
  for (int i = 0; i < nPredict; i++) {
    net->nnLayer[0]->nodes[i] = input[i];
  }

  for(int i = 1; i < net->nLayer; i++) {
    calcNodeValForLayer(net->nnLayer[i-1], net->nnLayer[i]);
  }
}

/*
util functions
*/
int trainDNN(int nPredict, neuralNet *net, const char pathToFile[]) {
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
    if (inpIt == nPredict) {
      // printf("\n ");
      // for(int j = 0; j < nPredict; j++) {
        // printf("%f ", inp[j]);
      // }
      feedForward(net, inp, nPredict);
      inpIt = 0;
      break;
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

int main(){
  int nPredict = 4;

  printf("nPredict: %d \n", nPredict);

  baseLayer *inpLayer = createLayer(nPredict, fullyConnected, reluActFunc);
  baseLayer *hiddenLayer1 = createLayer(8, fullyConnected, reluActFunc);
  baseLayer *outpLayer = createLayer(nPredict, fullyConnected, noActFunc);

  baseLayer *layer[] = {inpLayer, hiddenLayer1, outpLayer};
  neuralNet *dnn = createNet(layer, 3);

  int rc = trainDNN(nPredict, dnn, "../data/datasetByLine.csv");
  if (rc != 0) {
     printf("file open failed");
     return 0;
  }
  for (int i = 0; i < nPredict; i++) {
    printf("Last layer neuron num %i: %.2f \n", i, outpLayer->nodes[i]);
  }
}
