import numpy as np

# Activation function
def sigmoid(x):
    return 1.0/(1 + np.exp(-x))


# Derivative of activation function
def sigmoid_derivate(x):
    return x * (1.0 - x)


# NeuralNetwork Class
class NeuralNetwork:

    # For simplicity, kept biases 0
    def __init__(self,x,y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1],4)
        self.weights2 = np.random.rand(4,1)
        self.y = y
        self.output = np.zeros(self.y.shape)


    def feedForward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    
    # Applying chain rule to get the derivative of the loss function with respect to weights2 and weights1
    def backprop(self):
        d_weights2 = np.dot(self.layer1.T , (2*(self.y - self.output) * sigmoid_derivate(self.output)))
        d_weights1 = np.dot(self.input.T, (np.dot(2*(self.y - self.output) * sigmoid_derivate(self.output), self.weights2.T) * sigmoid_derivate(self.layer1)))
        self.weights1 += d_weights1
        self.weights2 += d_weights2


if __name__ == '__main__':
    X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
    y = np.array([[0],[1],[1],[0]])

    nn = NeuralNetwork(X,y)

    for i in range(1500):
        nn.feedForward()
        nn.backprop()
    
    
    print("Comparison between predicted and actual values:")
    print(np.concatenate((nn.output, nn.y),axis=1))