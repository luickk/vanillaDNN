#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>

// #define DEBUG 1

typedef void (*activationFunc)(bool derivative, float *inp, float *out);

typedef enum {
    fullyConnected,
} layerType;

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
void initLayer(float size, layerType type, baseLayer *layer, activationFunc actFunc) {
  layer->bias = (float*)malloc(size * sizeof(float));
  layer->weights = (float*)malloc(size * sizeof(float));
  layer->nodes = (float*)malloc(size * sizeof(float));
  layer->sensitives = (float*)malloc(size * sizeof(float));

  layer->actFunc = actFunc;
  layer->size = size;
  layer->type = type;
}

baseLayer *createLayer(float size, layerType type, activationFunc actFunc) {
  static baseLayer layer;
  initLayer(size, fullyConnected, &layer, actFunc);
  return &layer;
}

void setRandWeights(baseLayer *layer, int size){
  float randW = 0;
  for (int i = 0; i < size; i++) {
    randW = rand() % 10 + 1;
    layer->weights[i] = randW;
  }
}
void setRandNodes(baseLayer *layer, int size){
  float randN = 0;
  for (int i = 0; i < size; i++) {
    randN = rand() % 10 + 1;
    layer->nodes[i] = randN;
  }
}

void setRandBias(baseLayer *layer, int size){
  float randB = 0;
  for (int i = 0; i < size; i++) {
    randB = rand() % 10 + 1;
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
    setRandWeights(nn->nnLayer[i],nn->nnLayer[i]->size);
    setRandNodes(nn->nnLayer[i],nn->nnLayer[i]->size);
    setRandBias(nn->nnLayer[i],nn->nnLayer[i]->size);
  }

  return nn;
}

/*
Layer Operations
*/

// fast Sigmoid
// from https://stackoverflow.com/a/10733861
void sigmoidActFunc(bool derivative, float *x, float *y){
  if (!derivative) {
    *y = *x / (1 + fabsf(*x));
  } else {
    *y = *x / (1 + fabsf(*x));
    *y = *y * (1 - *y);
  }
}

void reluActFunc(bool derivative, float *x, float *y)
{
  if (!derivative) {
    *y = fmax(0, *x);
  } else {
    if (*x > 0) {
      *y = 1;
    } else if (*x <= 0){
      *y = 0;
    }
  }
}

void noActFunc(bool derivative, float *x, float *y)
{
  *y = *x;
}

void backpropagate(neuralNet *net, float *input, int learningRate) {
  float x, y = 0;
  // iterating over every neuron in output layer
  // calculating output (-> backprop init weights) weight

  // net->nLayer-1 = 0..nLastLayer
  baseLayer *lastLayer = net->nnLayer[net->nLayer-1];

  // last Layer L procedure differs from hidden layer backpropagation
  for (int i = 0; i < lastLayer->size; i++) {

    x = (lastLayer->weights[i] * input[i]);
    x += lastLayer->bias[i];
    lastLayer->actFunc(true, &x, &y);
    lastLayer->nodes[i] = (y * fabsf(input[i] - lastLayer->nodes[i]));

    lastLayer->weights[i] =  lastLayer->nodes[i] * lastLayer->sensitives[i];
    lastLayer->bias[i] =  lastLayer->nodes[i] * lastLayer->sensitives[i];

    lastLayer->nodes[i] = lastLayer->nodes[i] * lastLayer->weights[i];
    lastLayer->nodes[i] += lastLayer->bias[i];

    #ifdef DEBUG
      printf("backprop - Last Layer neuron: %i, weight: %.0f \n", i, lastLayer->weights[i]);
    #endif
  }

  // -1, without last layer
  for(int i = net->nLayer-1; i == 0; i--) {
    for (int j = 0; j<net->nnLayer[i]->size; j++) {

      x = (net->nnLayer[i+1]->weights[j] * net->nnLayer[i+1]->nodes[j]);
      // printf("x: %f", x);
      x += net->nnLayer[i+1]->bias[j];

      net->nnLayer[i]->actFunc(true, &x, &y);
      net->nnLayer[i]->sensitives[j] = y * net->nnLayer[i+1]->weights[j]*net->nnLayer[i+1]->sensitives[j];

      net->nnLayer[i]->weights[j] = (learningRate * net->nnLayer[i]->sensitives[j]) * net->nnLayer[i+1]->nodes[i];
      net->nnLayer[i]->bias[j] = (learningRate * net->nnLayer[i]->sensitives[j]) * net->nnLayer[i+1]->bias[i];

      net->nnLayer[i]->nodes[i] = net->nnLayer[i]->nodes[i] * net->nnLayer[i]->weights[i];
      net->nnLayer[i]->nodes[i] += net->nnLayer[i]->bias[i];

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
      net->nnLayer[l]->actFunc(false, &net->nnLayer[l]->nodes[i], &net->nnLayer[l]->nodes[i]);
    }
  }
}

void lsErrorCalc(neuralNet *net, float *input, float *error) {
  *error = 0;
  for (int i = 0; i < net->nnLayer[net->nLayer-1]->size; i++) {
    *error += 0.5*sqrt(fabsf(input[i] - net->nnLayer[net->nLayer-1]->nodes[i]));
  }
  #ifdef DEBUG
    printf("%f,", *error);
  #endif
}

/*
util functions
*/
int trainDNN(neuralNet *net, int nPredict, const char pathToFile[], int iterations, int learningRate) {
  FILE *fp;
  char *line = 0;
  int inpIt = 0;
  int nLines = 0;
  size_t len = 0;
  ssize_t read;

  float *inp = (float*)malloc(nPredict*sizeof(float));
  float error = 0;
  float meanErr = 0;

  printf("mean Err: ");
  for (int i = 0; i < iterations; i++) {
    fp = fopen(pathToFile, "r");
    if (fp == 0) {
     return 1;
    }
    while ((read = getline(&line, &len, fp)) != -1) {
      if (inpIt == nPredict) {

        backpropagate(net, inp, learningRate);
        lsErrorCalc(net, inp, &error);

        meanErr += error;
        nLines++;
        inpIt = 0;
      }
      inp[inpIt] = atof(line);
      inpIt++;
    }
    meanErr = meanErr/nLines;
    printf("%f,", meanErr);
    nLines = 0;
    meanErr = 0;

    fclose(fp);
  }
  printf("\n");
  if (line) {
   free(line);
  }
  return 0;
}

int predictDNN(neuralNet *net, float *predictionSeq) {
  feedForward(net, predictionSeq);
  baseLayer *lastLayer = net->nnLayer[net->nLayer-1];

  for (int i = 0; i < lastLayer->size; i++) {
    printf("Prediction %i, node val: %f \n", i, lastLayer->nodes[i]);
  }
  return 0;
}

int main(){
  int nPredict = 4;
  int iterations = 100;
  int learningRate = 1;

  #ifdef DEBUG
    printf("nPredict: %d \n", nPredict);
  #endif

  baseLayer *inpLayer = createLayer(nPredict, fullyConnected, reluActFunc);
  baseLayer *hiddenLayer1 = createLayer(8, fullyConnected, reluActFunc);
  baseLayer *outpLayer = createLayer(nPredict, fullyConnected, reluActFunc);

  baseLayer *layer[] = {inpLayer, hiddenLayer1, outpLayer};
  neuralNet *dnn = createNet(layer, 3);

  int rc = trainDNN(dnn, nPredict, "../data/datasetByLine.csv", iterations, learningRate);

  for (int i = 0; i < nPredict; i++) {
    printf("Last layer neuron %i, node val: %.2f \n", i, outpLayer->nodes[i]);
  }

  float predSeq[] = {2.6, 2.4, 3.9,  1.3, 2.1};
  predictDNN(dnn, predSeq);
}
