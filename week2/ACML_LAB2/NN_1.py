import torch

class MultiLayerPerceptron():
    def __init__(self, input_size, layer_sizes):
        self.layers = len(layer_sizes)
        self.weights = []
        self.biases = []
        self.activations = []

        prev_size = input_size
        for size in layer_sizes:
            self.weights.append(torch.randn(prev_size, size))
            self.biases.append(torch.randn(size))
            prev_size = size

    def forward(self, input, index):
        a = torch.add(torch.matmul(input, self.weights[index]), self.biases[index])
        a = torch.sigmoid(a)
        self.activations.append(a)

input_size = 4
layer_sizes = [4, 2]
input = torch.randn(input_size)

mlp = MultiLayerPerceptron(input_size, layer_sizes)

for (_,index) in enumerate(layer_sizes):
    mlp.forward(input, index)



