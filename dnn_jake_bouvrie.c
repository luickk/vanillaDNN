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
  baseLayer **nnLayer;
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

baseLayer createLayer(int size, layerType type, activationFunc actFunc) {
  baseLayer *layer = (baseLayer *)malloc(sizeof(baseLayer));
  initLayer(size, fullyConnected, layer, actFunc);
  return *layer;
}

void setRandWeights(baseLayer *layer, int size){
  float randW = 0;
  for (int i = 0; i < size; i++) {
    randW = rand() % 10 + 1;
    layer->weights[i] = randW/10;
  }
}
void setRandNodes(baseLayer *layer, int size){
  float randN = 0;
  for (int i = 0; i < size; i++) {
    randN = rand() % 10 + 1;
    layer->nodes[i] = randN/10;
  }
}

void setRandBias(baseLayer *layer, int size){
  float randB = 0;
  for (int i = 0; i < size; i++) {
    randB = rand() % 10 + 1;
    layer->bias[i] = randB/10;
  }
}

neuralNet createNet(baseLayer *layer[], int nLayer) {
  neuralNet *nn = (neuralNet *)malloc(sizeof(baseLayer)*nLayer+sizeof(nLayer));

  nn->nnLayer=layer;

  nn->nLayer = nLayer;

  for (int i=0; i<nLayer; ++i) {
    setRandWeights(nn->nnLayer[i],nn->nnLayer[i]->size);
    setRandNodes(nn->nnLayer[i],nn->nnLayer[i]->size);
    setRandBias(nn->nnLayer[i],nn->nnLayer[i]->size);
  }

  return *nn;
}

void freeNet(neuralNet *net) {
  for(int i = 0; i < net->nLayer; i++) {
    free(net->nnLayer[i]->bias);
    free(net->nnLayer[i]->weights);
    free(net->nnLayer[i]->nodes);
    free(net->nnLayer[i]->sensitives);
    free(net->nnLayer[i]);
  }
  free(&net);
}

void printNN(neuralNet *net) {
  printf("------- nn ------- \n");
  for(int i = 0; i < net->nLayer; i++) {
    printf("Layer %i: ", i);
    for (int j = 0; j<net->nnLayer[i]->size; j++) {
      printf("%f ", net->nnLayer[i]->nodes[j]);
    }
    printf("\n");
  }
  printf("------- nn ------- \n");
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

void leakyReluActFunc(bool derivative, float *x, float *y)
{
  if (!derivative) {
    *y = fmax(0.1* *x, *x);
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

void backpropagate(neuralNet *net, float *input, float learningRate) {
  float x, y = 0;
  // iterating over every neuron in output layer
  // calculating output (-> backprop init weights) weight

  // net->nLayer-1 = 0..nLastLayer
  baseLayer *lastLayer = net->nnLayer[net->nLayer-1];

  printf("-----------------------------------------\n");
  // last Layer L procedure differs from hidden layer backpropagation
  for (int i = 0; i < lastLayer->size; i++) {
    printf("-----------\n");
    x = (lastLayer->weights[i] * input[i]);
    printf("x: %f \n", x);
    // x += lastLayer->bias[i];
    printf("x+b: %f \n", x);
    lastLayer->actFunc(true, &x, &y);
    printf("actFunc: %f \n", x);
    printf("difference: %f - %f = %f \n", lastLayer->nodes[i], input[i], (lastLayer->nodes[i]-input[i]));
    lastLayer->sensitives[i] = (y * (lastLayer->nodes[i] - input[i]));
    lastLayer->sensitives[i] = (learningRate*lastLayer->sensitives[i]);
    printf("sensitives: %f \n", lastLayer->sensitives[i]);

    // weight update by sens sub
    // lastLayer->weights[i] = input[i] - lastLayer->sensitives[i];
    // weight update by sens mult
    lastLayer->weights[i] =  lastLayer->sensitives[i] * input[i];

    printf("weights: %f \n", lastLayer->weights[i]);
    lastLayer->bias[i] =  lastLayer->sensitives[i] * input[i];
    printf("bias: %f \n", lastLayer->bias[i]);

    x = lastLayer->nodes[i] * lastLayer->weights[i];
    // x += lastLayer->bias[i];
    printf("x: %f \n", x);
    lastLayer->actFunc(false, &x, &y);
    printf("final act func: %f \n", y);

    lastLayer->nodes[i] = y;
    printf("-----------\n");

    #ifdef DEBUG
      printf("backprop - Last Layer neuron: %i, weight: %.0f \n", i, lastLayer->weights[i]);
    #endif
  }
  printf("-----------------------------------------\n");

  // -2, without last layer
  for(int i = net->nLayer-2; i >= 0; i--) {
    printf("-----------------------------------------\n");
    for (int j = 0; j<net->nnLayer[i]->size; j++) {
      printf("-----------\n");
      x = (net->nnLayer[i]->weights[j] * net->nnLayer[i+1]->nodes[j]);
      printf("x: %f \n", x);
      // x += net->nnLayer[i]->bias[j];
      printf("x+b: %f \n", x);

      net->nnLayer[i]->actFunc(true, &x, &y);
      printf("act func x: %f \n", y);
      net->nnLayer[i]->sensitives[j] = y * net->nnLayer[i+1]->weights[j]*net->nnLayer[i+1]->sensitives[j];
      net->nnLayer[i]->sensitives[j] = (learningRate * net->nnLayer[i]->sensitives[j]);
      printf("sensitives: %f \n", net->nnLayer[i]->sensitives[j]);

      net->nnLayer[i]->weights[j] = net->nnLayer[i]->sensitives[j] * net->nnLayer[i+1]->nodes[j];
      printf("weight: %f \n", net->nnLayer[i]->weights[j]);
      net->nnLayer[i]->bias[j] = net->nnLayer[i]->sensitives[j] * net->nnLayer[i+1]->nodes[j];
      printf("bias: %f \n", net->nnLayer[i]->bias[j]);



      x = net->nnLayer[i]->nodes[j] * net->nnLayer[i]->weights[j];
      // x += net->nnLayer[i]->bias[j];

      printf("x: %f \n", x);
      net->nnLayer[i]->actFunc(false, &x, &y);

      printf("final act func: %f \n", y);
      net->nnLayer[i]->nodes[j] = y;
      printf("-----------\n");

      #ifdef DEBUG
        printf("backprop - Layer: %i, neuron: %i, weight: %.0f \n", i, j, net->nnLayer[i]->weights[j]);
      #endif
    }
    printf("-----------------------------------------\n");
  }
  printNN(net);
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
    // printf("%f, %f \n", input[i], net->nnLayer[net->nLayer-1]->nodes[i]);
    *error += 0.5*sqrt(input[i] - net->nnLayer[net->nLayer-1]->nodes[i]);
  }
  #ifdef DEBUG
    printf("%f,", *error);
  #endif
}


/*
util functions
*/
int trainDNN(neuralNet *net, int nPredict, const char pathToFile[], int iterations, float learningRate) {
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

// TODO -> free memory!!
int main(){
  int nPredict = 4;
  int nLayer = 0;
  int iterations = 1;
  float learningRate = 0.1;

  #ifdef DEBUG
    printf("nPredict: %d \n", nPredict);
  #endif

  baseLayer inpLayer = createLayer(nPredict, fullyConnected, leakyReluActFunc);
  baseLayer hiddenLayer1 = createLayer(16, fullyConnected, leakyReluActFunc);
  baseLayer hiddenLayer2 = createLayer(8, fullyConnected, leakyReluActFunc);
  baseLayer outpLayer = createLayer(nPredict, fullyConnected, leakyReluActFunc);

  nLayer = 4;
  baseLayer **layer = (baseLayer**)malloc(nLayer*sizeof(baseLayer));
  layer[0] = &inpLayer;
  layer[1] = &hiddenLayer1;
  layer[2] = &hiddenLayer2;
  layer[3] = &outpLayer;

  neuralNet dnn = createNet(layer, nLayer);

  int rc = trainDNN(&dnn, nPredict, "../data/datasetByLine.csv", iterations, learningRate);

  for (int i = 0; i < nPredict; i++) {
    printf("Last layer neuron %i, node val: %.2f \n", i, outpLayer.nodes[i]);
  }

  // float predSeq[] = {2.6, 2.4, 3.9,  1.3, 2.1};
  float predSeq[] = {14.6, 18.2, 16.4, 16.6, 14.7};
  printNN(&dnn);
  predictDNN(&dnn, predSeq);

  freeNet(&dnn);
}
