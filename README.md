# Vanilla CNN

The goal of the project is to build a vanilla dnn/cnn from scratch with no help from any kind of libraries (except from std C) in a performant and memory sparing C lang context.
The code itself is not meant to be used as library but could(if required) easily be reused as such since all functions are isolated from the execution context.

## DNN

The `dnn.cpp` contains functions to build a simple neural net with a means Square error reduction and interchangeable activation functions. To test the NN it predicts n numbers from a training dataset.
