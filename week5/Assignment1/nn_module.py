import numpy as np
import argparse
class MultiLayerPerceptron:
    def __init__(self, input_size, hidden_size, output_size):
        self.lr = 0.1
        self.weights1 = np.ones((input_size, hidden_size))
        self.biases1 = np.ones(hidden_size)
        self.weights2 = np.ones((hidden_size, output_size))
        self.biases2 = np.ones(output_size)
        self.hidden = np.array([])
        self.output = np.array([])


    def forward(self, x):
        # activations
        self.hidden = self.sigmoid(np.matmul(x, self.weights1) + self.biases1)
        self.output = self.sigmoid(np.matmul(self.hidden, self.weights2) + self.biases2)

    @staticmethod
    def sigmoid(s):
        return 1 / (1 + np.exp(-s))

    @staticmethod
    def tail(s):
        return s * (1 - s)

    def backward(self, x, label):
        # deltas
        self.delta_output = ((self.output - label) * self.tail(self.output))
        self.delta_hidden = np.matmul(self.weights2, self.delta_output) * self.tail(self.hidden)


        # weights and biases
        self.weights1 -= np.outer(x, self.delta_hidden) * self.lr
        self.biases1 -= self.delta_hidden * self.lr

        self.weights2 -= np.outer(self.hidden, self.delta_output) * self.lr
        self.biases2 -= self.delta_output * self.lr

    def compute_loss(self, label):
        return 0.5 * np.sum(np.square(self.output - label))


input_size = 4
hidden_size = 8
output_size = 3

input_numbers = [float(input()) for i in range(7)]
inputs = input_numbers[:4]
labels = input_numbers[4:]

# Separate the inputs and labels based on the single input argument
mlp = MultiLayerPerceptron(input_size, hidden_size, output_size)

mlp.forward(inputs)
loss = mlp.compute_loss(labels)
print(round(loss, 4))

# update
mlp.backward(inputs, labels)

mlp.forward(inputs)
loss = mlp.compute_loss(labels)
print(round(loss, 4))