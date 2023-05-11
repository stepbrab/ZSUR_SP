import matplotlib.pyplot as plt
import numpy as np


class Perceptron:
    def __init__(self, num_inputs=2, num_outputs=3, learning_rate=0.1):
        self.weights = np.random.randn(num_inputs, num_outputs)
        self.bias = np.zeros(num_outputs)
        self.learning_rate = learning_rate
        print('Spouští se Perceptron...')

    def sigmoid(self, x):
        return 2 / (1 + np.exp(-x))-1

    def predict(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        activation = self.sigmoid(weighted_sum)
        return np.argmax(activation, axis=1)

    def train_batch_gd(self, inputs, targets, num_epochs, batch_size=None):
        for epoch in range(num_epochs):
            if batch_size is None:
                batch_size = inputs.shape[0]
            for i in range(0, inputs.shape[0], batch_size):
                batch_inputs = inputs[i:i + batch_size]
                batch_targets = targets[i:i + batch_size]
                activation = self.sigmoid(np.dot(batch_inputs, self.weights) + self.bias)
                gradients = np.dot(batch_inputs.T, activation - np.eye(self.weights.shape[1])[batch_targets])
                biases_grad = np.sum(activation - np.eye(self.bias.shape[0])[batch_targets], axis=0)
                self.weights -= self.learning_rate * gradients / batch_inputs.shape[0]
                self.bias -= self.learning_rate * biases_grad / batch_inputs.shape[0]

    def train_sgd(self, inputs, targets, num_epochs):
        for epoch in range(num_epochs):
            for i in np.random.permutation(inputs.shape[0]):
                y = self.predict(inputs[i].reshape(1, -1))
                if y != targets[i]:
                    activation = self.sigmoid(np.dot(inputs[i], self.weights) + self.bias)
                    gradient = np.outer(inputs[i], activation - np.eye(self.weights.shape[1])[targets[i]])
                    biases_grad = activation - np.eye(self.bias.shape[0])[targets[i]]
                    self.weights -= self.learning_rate * gradient
                    self.bias -= self.learning_rate * biases_grad

    def plot_data(self, inputs, targets):
        colors = ['r', 'g', 'b']
        for i in range(targets.max() + 1):
            plt.scatter(inputs[:, 0][targets == i], inputs[:, 1][targets == i], color=colors[i], label=f"Shluk {i + 1}")
        plt.xlabel('x')
        plt.ylabel('y')

    def plot_boundary(self, inputs, targets, title='Perceptron'):
        plt.figure(figsize=(8, 8))

        min_values = np.min(inputs, axis=0)
        max_values = np.max(inputs, axis=0)
        xx, yy = np.meshgrid(np.linspace(min_values[0], max_values[0], 100),
                             np.linspace(min_values[1], max_values[1], 100))

        mesh_inputs = np.c_[xx.ravel(), yy.ravel()]
        predictions = self.predict(mesh_inputs)
        Z = predictions.reshape(xx.shape)

        for i in range(targets.max() + 1):
            plt.scatter(inputs[:, 0][targets == i], inputs[:, 1][targets == i], label=f"Shluk {i + 1}")

        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.title(title)
        plt.show()
