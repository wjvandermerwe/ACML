import torch

class MultiLayerPerceptron():
    def __init__(self, input_size, hidden_size, output_size):
        self.lr = 0.01
        # for each tendril
        self.weights1 = torch.randn(input_size, hidden_size)
        self.biases1 = torch.randn(hidden_size)
        self.weights2 = torch.randn(hidden_size, output_size)
        self.biases2 = torch.randn(output_size)

    def forward(self, X):
        # activations
        self.hidden = torch.sigmoid(torch.matmul(X, self.weights1) + self.biases1)
        self.output = torch.sigmoid(torch.matmul(self.hidden, self.weights2) + self.biases2)
        return self.output

    def tail(self, s):
        return s * (1 - s)
    def backward(self, X, Y, output):
        # deltas
        self.delta_output = Y - output * self.tail(output)
        self.delta_hidden = torch.matmul(self.delta_output, self.weights2.T) * self.tail(self.hidden)

        # weights and biases
        self.weights1 += torch.matmul(X.T, self.delta_hidden) * self.lr
        self.weights2 += torch.matmul(self.hidden.T, self.delta_output) * self.lr

        self.biases1 += torch.sum(self.delta_hidden) * self.lr
        self.biases2 += torch.sum(self.delta_output) * self.lr

    def compute_loss(self, Y, output):
        return torch.sum(0.5 * torch.square(Y-output)**2)

    def train(self, X, Y):
        output = self.forward(X)
        self.backward(X, Y, output)
        return self.compute_loss(Y, output)

    def log_output(self, iteration, loss):
        print(f"Iteration: {iteration}, Loss: {loss}")

input_size = 4
hidden_size = 4
output_size = 2

mlp = MultiLayerPerceptron(input_size, hidden_size, output_size)
inputs = torch.randn(20, 4)
labels = torch.randn(20, 2)

max_iterations = 5000
log_interval = 20

loss = 1
iteration = 0
while loss > 0 and max_iterations > iteration:
    loss = mlp.train(inputs, labels)
    if iteration % log_interval == 0:
        mlp.log_output(iteration, loss)
    iteration += 1