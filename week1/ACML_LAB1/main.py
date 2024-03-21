import numpy as np
import matplotlib.pyplot as plt

points = np.arange(4 * 2).reshape(-1, 2)
diagonal_points = np.vstack((points[:, :1], points[:, ::2]))
class Perceptron:
    def __init__(self, input_size, lr=0.01):
        self.weights = np.zeros(input_size + 1)  # +1 for the bias
        self.lr = lr
    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return 1 if summation > 0 else 0

    def train(self, training_inputs, labels):
        for _ in enumerate(training_inputs):
            indices = np.arange(training_inputs.shape[0])
            np.random.shuffle(indices)
            training_inputs = training_inputs[indices]
            labels = labels[indices]

            for input, label in zip(training_inputs, labels):
                prediction = self.predict(input)
                self.weights[1:] += self.lr * (label - prediction) * input
                self.weights[0] += self.lr * (label - prediction)

    def compute_loss(self, inputs, labels):
        outputs = np.array([self.predict(input) for input in inputs])
        result = np.sum(np.abs(labels - outputs))
        return result

    def log_model_details(self):
        print("Model Weights:", self.weights)
        print("Learning Rate:", self.lr)


np.random.seed(0)

inputs = np.random.uniform(low=1, high=5, size=(100, 2))

def f(x1, x2):
    return 2*x1 + 3*x2 - 1

labels = np.array([1 if f(x1, x2) > 0 else 0 for x1, x2 in inputs])

perceptron = Perceptron(2)


max_iterations = 200
iterations = 0
while perceptron.compute_loss(inputs, labels) > 0 and iterations < max_iterations:
    perceptron.train(inputs, labels)
    iterations += 1

perceptron.log_model_details()


w = perceptron.weights

x1 = np.linspace(1, 5, 100)
x2 = -(w[1] / w[2]) * x1 - (w[0] / w[2])

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x1, x2, label='Linear Discriminant (Decision Boundary)')
plt.scatter(inputs[:, 0], inputs[:, 1], c=labels, cmap='bwr', marker='o', label='Data Points')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Perceptron Decision Boundary')
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.show()