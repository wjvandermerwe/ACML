import numpy as np


class MultiLayerPerceptron:
    def __init__(self, input_size, hidden_size, output_size):
        self.lr = 0.1
        self.weights1 = np.ones((input_size, hidden_size))
        self.biases1 = np.ones(hidden_size)
        self.weights2 = np.ones((hidden_size, output_size))
        self.biases2 = np.ones(output_size)

    def forward(self, x):
        print("weights1:",self.weights1)
        print("weights2:",self.weights2)

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
        self.delta_hidden = np.matmul(self.delta_output, self.weights2.T) * self.tail(self.hidden)


        # weights and biases
        self.weights1 -= np.outer(x, self.delta_hidden) * self.lr
        self.weights2 -= np.outer(self.hidden, self.delta_output) * self.lr
        self.biases1 -= np.sum(self.delta_hidden) * self.lr
        self.biases2 -= np.sum(self.delta_output) * self.lr

    def compute_loss(self, label):
        return 0.5 * np.sum(np.square(self.output - label))


input_size = 4
hidden_size = 8
output_size = 3

# inputs = np.array([-2, 1, 0.5, -1])
# labels = np.array([0, 0, 1])
inputs = np.array([1.4, 0, -2.5, -3])
labels = np.array([0.4, 0.6, 0])
mlp = MultiLayerPerceptron(input_size, hidden_size, output_size)


mlp.forward(inputs)
loss = mlp.compute_loss(labels)
print(f"1 Loss: {loss}")

mlp.backward(inputs, labels)
mlp.forward(inputs)
loss = mlp.compute_loss(labels)
print(f"2 Loss: {loss}")