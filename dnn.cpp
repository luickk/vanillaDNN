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
  float *sensitives;
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
  layer.sensitives = (float*)malloc(size * sizeof(float));

  layer.actFunc = actFunc;
  layer.size = size;
  layer.type = type;
}

baseLayer *createLayer(float size, layerType type, activationFunc actFunc) {
  static baseLayer layer;
  initLayer(size, fullyConnected, layer, actFunc);
  return &layer;
}

void setRandWeights(baseLayer *layer, int size){
  int randW = 0;
  for (int i = 0; i < size; i++) {
    randW = (rand() % (1 - 0 + 0)) + 0;
    layer->weights[i] = randW/100;
  }
}

void setRandBias(baseLayer *layer, int size){
  int randB = 0;
  for (int i = 0; i < size; i++) {
    randB = (rand() % (1 - 0 + 0)) + 0;
    layer->bias[i] = randB/100;
  }
}

neuralNet *createNet(baseLayer *layer[], int nLayer) {
  neuralNet *nn = (neuralNet *)malloc(sizeof(baseLayer)*nLayer+sizeof(nLayer));
  for (int i=0; i<nLayer; ++i) {
     nn->nnLayer[i]=layer[i];
  }
  nn->nLayer = nLayer;

  for (int i=0; i<nLayer; ++i) {
    setRandWeights(nn->nnLayer[i],nn->nnLayer[i]->size);
    setRandBias(nn->nnLayer[i],nn->nnLayer[i]->size);
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

void backpropagate(neuralNet *net, float *input) {
  float y = 0;
  // iterating over every neuron in output layer
  // calculating output (-> backprop init weights) weight

  // net->nLayer-1 = 0..nLastLayer
  baseLayer *lastLayer = net->nnLayer[net->nLayer-1];

  for (int i = 0; i < net->nnLayer[net->nLayer-1]->size; i++) {
    reluDerivativeActFunc(&lastLayer->nodes[i], &y);
    lastLayer->sensitives[i] = (y * (input[i] - lastLayer->nodes[i]));
    lastLayer->weights[i] =  lastLayer->nodes[i] * lastLayer->sensitives[i];
    #ifdef DEBUG
      printf("backprop - Last Layer neuron: %i, weight: %.0f \n", i, lastLayer->weights[i]);
    #endif
  }

  float neuronInputY, neuronInputX = 0;
  for(int i = net->nLayer-2; i == 0; i--) {
    for (int j = 0; j<net->nnLayer[i]->size; j++) {

      neuronInputX = (net->nnLayer[i]->weights[j] * net->nnLayer[i]->nodes[j+1]);
      neuronInputX += net->nnLayer[i]->bias[j];

      reluDerivativeActFunc(&neuronInputX, &neuronInputY);
      net->nnLayer[i]->sensitives[j] = neuronInputY * net->nnLayer[i]->weights[j+1]*net->nnLayer[i]->sensitives[j+1];

      net->nnLayer[i]->weights[j] = net->nnLayer[i]->sensitives[j] * net->nnLayer[i]->nodes[i+1];

      #ifdef DEBUG
        printf("backprop - Layer: %i, neuron: %i, weight: %.0f \n", i, j, net->nnLayer[i]->weights[j]);
      #endif
    }
  }
}

void feedForward(neuralNet *net, float *input){
  // setting input to input layer nodes
  for (int i = 0; i < net->nnLayer[0]->size; i++) {
    net->nnLayer[0]->nodes[i] = input[i];
  }

  // iterating over every layer
  for(int l = 1; l < net->nLayer; l++) {
    // iterating over every neuron in layer l
    for(int i = 0; i<net->nnLayer[l]->size; i++) {
      // iterating over every neuron in layer l-1
      for (int j = 0; j<net->nnLayer[l-1]->size; j++) {
        net->nnLayer[l]->nodes[i] += net->nnLayer[l]->weights[i] * net->nnLayer[l-1]->nodes[j];
        #ifdef DEBUG
          printf("ff - Layer: %i, neuron: %i, node: %.0f \n", i, j, net->nnLayer[l]->nodes[i]);
        #endif
      }
      net->nnLayer[l]->nodes[i] += net->nnLayer[l]->bias[i];
      net->nnLayer[l]->actFunc(&net->nnLayer[l]->nodes[i], &net->nnLayer[l]->nodes[i]);
    }
  }
}

void lsErrorCalc(neuralNet *net, float *input, float *error) {
  *error = 0;
  for (int i = 0; i < net->nnLayer[net->nLayer-1]->size; i++) {
    *error += 0.5*sqrt(abs(input[i] - net->nnLayer[net->nLayer-1]->nodes[i]));
  }
  #ifdef DEBUG
    printf("%f,", *error);
  #endif
}

/*
util functions
*/
int trainDNN(int nPredict, neuralNet *net, const char pathToFile[], int iterations) {
  FILE *fp;
  char *line = 0;
  int inpIt = 0;
  size_t len = 0;
  ssize_t read;

  float *inp = (float*)malloc(nPredict*sizeof(float));
  float error = 0;

  fp = fopen(pathToFile, "r");
  if (fp == 0) {
   return 1;
  }
  for (int i = 0; i < iterations; i++) {
    while ((read = getline(&line, &len, fp)) != -1) {
      if (inpIt == nPredict) {

        feedForward(net, inp);
        backpropagate(net, inp);
        lsErrorCalc(net, inp, &error);
        printf("%.0f,", error);

        inpIt = 0;
      }
      inp[inpIt] = atof(line);
      inpIt++;
    }
  }
  printf("\n");
  fclose(fp);
  if (line) {
   free(line);
  }
  return 0;
}

int main(){
  int nPredict = 20;
  int iterations = 1;
  #ifdef DEBUG
    printf("nPredict: %d \n", nPredict);
  #endif


  baseLayer *inpLayer = createLayer(nPredict, fullyConnected, reluActFunc);
  baseLayer *hiddenLayer1 = createLayer(8, fullyConnected, reluActFunc);
  baseLayer *outpLayer = createLayer(nPredict, fullyConnected, noActFunc);

  baseLayer *layer[] = {inpLayer, hiddenLayer1, outpLayer};
  neuralNet *dnn = createNet(layer, 3);

  int rc = trainDNN(nPredict, dnn, "../data/datasetByLine.csv", iterations);
  if (rc != 0) {
     printf("file open failed");
     return 0;
  }
  for (int i = 0; i < nPredict; i++) {
    printf("Last layer neuron %i, node val: %.2f \n", i, outpLayer->nodes[i]);
  }
}
