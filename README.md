# Vanilla DNN

The goal of the project is to build a vanilla dnn/cnn from scratch with no help from any kind of libraries (except from std C) in a performant and memory saving C lang context.
The code itself is not meant to be used as a library but could(if required) easily be reused as such since all functions are isolated from the execution context.

## DNN by Jake Bouvrie

The `dnn_jake_bouvrie.cpp` contains functions to build a simple neural net with a means Square error reduction and interchangeable activation functions. To test the NN it predicts n numbers from a training dataset.

The mathematical implementation can be found [here](http://www.cogprints.org/5869/1/cnn_tutorial.pdf) and was written by Jake Bouvrie.

When training the model the neural net faces a problem called "dying relu", which is caused by a great difference between the output layer nodes and the true output, that leads to a overcorrection which then triggers the activation function and sets the nodes val to 0. The problem is a question of parameter optimisation and as such a todo for the Jake Bouvrie nn.

## Simple DNN

`dnn_simple.cpp` is very similar to other one but has a more simple backpropagation/ weight update algorithm which makes optimisation easier. The results of this nn are plausible although there is certainly a lot of room for (parameter, dataset)optimisation.
