# Perceptron
 
A "Perceptron" is simply a mathematical function that takes a set of inputs, performs a mathematical computation and returns an output. 

The project is an implementation of a single layer neural network without using any external libraires to gain a intuitive understanding of the inner workings of a Neural Networks. 

## Neural Networks
A typical neural network consists of the following components:
* An input layer, *x*
* A number of hidden layers
* An outpout layer, *y*
* A set of weights and baises between each layer (*W, b*)
* An activation function for each hidden layer, \sigma

The process of fine tuning the weights and biases of a neural network from the input data is known as training the neural network.

Each iteration of the training process consists of the following steps:
* Calculating the predicted output, known as FeedForward
* Updating weights and biases, known as Backpropogation

The goal of training a neural network is to minimise the loss function, in this example, *Sum-of-Sqaures Error* is chosen as the loss function.