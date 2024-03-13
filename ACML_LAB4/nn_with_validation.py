import torch
from dataset import validation_inputs, validation_labels, training_inputs, training_labels, test_inputs, test_labels


class MultiLayerPerceptron():
    def __init__(self, input_size, hidden_size, output_size):
        self.lr = 0.0001
        self.lmbd = 0.00000001

        # for each tendril
        self.weights1 = torch.randn(input_size, hidden_size)
        self.biases1 = torch.randn(hidden_size)
        self.weights2 = torch.randn(hidden_size, output_size)
        self.biases2 = torch.randn(output_size)

    def forward(self, X):
        X = X.float()

        # activations
        self.hidden = torch.sigmoid(torch.matmul(X, self.weights1) + self.biases1)
        self.output = torch.softmax(torch.matmul(self.hidden, self.weights2) + self.biases2, dim=1)

        return self.output


    def normalise_data(self, inputs):
        return (inputs - inputs.min()) / (inputs.max() - inputs.min())

    def tail(self, s):
        return s * (1 - s)
    def backward(self, X, Y, output):
        X = X.float()
        # deltas
        self.delta_output = ((output - Y) * self.tail(output)).float()
        self.delta_hidden = torch.matmul(self.delta_output, self.weights2.T) * self.tail(self.hidden)

        # weights and biases -> L2 regularized
        self.weights1 -= (torch.matmul(X.T, self.delta_hidden) * self.lr -
                          (self.lr * self.lmbd * self.weights1))

        self.weights2 -= (torch.matmul(self.hidden.T, self.delta_output) * self.lr -
                          (self.lr * self.lmbd * self.weights2))

        self.biases1 -= torch.sum(self.delta_hidden) * self.lr
        self.biases2 -= torch.sum(self.delta_output) * self.lr

    # def update_learn_rate(self, max_iterations, iteration):
        # self.lr -= 0.0009*(iteration/max_iterations)

    def compute_loss(self, labels, output):

        # sum squares error
        # return torch.sum(0.5 * torch.square(Y-output)**2)

        # cross entropy loss
        # return torch.nn.functional.cross_entropy()
        return -torch.sum(labels * torch.log(output))

    def train(self, X, Y):
        X = self.normalise_data(X)
        output = self.forward(X)
        self.backward(X, Y, output)
        return self.compute_loss(Y, output)

    def log_output(self, iteration, loss, train_loss):
        print(f"Iteration: {iteration}, Validation Loss: {loss}, Training Loss: {train_loss}")


mlp = MultiLayerPerceptron(input_size= 7, hidden_size = 7, output_size=2)

previous_validation_loss = float('inf')
max_iterations = 100000
log_interval = 1000
loss = 1
patience = 3
iteration = 0
no_improvement_count = 0

while True and max_iterations > iteration:
    validation_indices = torch.arange(len(validation_inputs))
    validation_shuffled_indices = torch.randperm(len(validation_indices))

    indices = torch.arange(len(training_inputs))
    shuffled_indices = torch.randperm(len(indices))
    # Use the shuffled indices to reorder both tensors
    validation_loss = mlp.train(validation_inputs[validation_shuffled_indices], validation_labels[validation_shuffled_indices])
    loss = mlp.train(training_inputs[shuffled_indices], training_labels[shuffled_indices])
    if iteration % log_interval == 0:
        mlp.log_output(iteration, validation_loss, loss)


    if validation_loss >= previous_validation_loss:
        no_improvement_count += 1
        if no_improvement_count >= patience:
            print("Stopping training due to no improvement in validation loss.")
            break
    else:
        # Reset the no_improvement_count if validation loss improved
        no_improvement_count = 0

    if validation_loss < previous_validation_loss:
        previous_validation_loss = validation_loss

    # mlp.update_learn_rate(max_iterations, iteration)

    iteration += 1

# predict_input = torch.tensor([5.1000, 3.5000, 1.4000, 0.2000]).unsqueeze(0)
# output_tensor = mlp.forward(predict_input)
#
# output_values = output_tensor.squeeze().tolist()
#
# # Convert scientific notation to normal decimal form
# output_values_normal = [f'{val:.10f}' for val in output_values]
#
# print(output_values_normal)