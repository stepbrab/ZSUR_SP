import matplotlib.pyplot as plt
import numpy as np


class Perceptron:
    def __init__(self, num_inputs, num_outputs, learning_rate=0.1, topology='single'):
        self.weights = np.random.randn(num_inputs, num_outputs)
        self.bias = np.zeros(num_outputs)
        self.learning_rate = learning_rate
        self.topology = topology

    def predict(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return np.argmax(weighted_sum)

    def predict_multi(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return weighted_sum

    def train_batch_gd(self, inputs, targets, num_epochs, batch_size=None):
        for epoch in range(num_epochs):
            if batch_size is None:
                batch_size = inputs.shape[0]
            for i in range(0, inputs.shape[0], batch_size):
                batch_inputs = inputs[i:i + batch_size]
                batch_targets = targets[i:i + batch_size]
                predictions = np.argmax(np.dot(batch_inputs, self.weights) + self.bias, axis=1)
                gradients = np.zeros(self.weights.shape)
                biases_grad = np.zeros(self.bias.shape)
                for j in range(batch_inputs.shape[0]):
                    gradients += np.outer(batch_inputs[j], np.eye(self.weights.shape[1])[batch_targets[j]] -
                                          np.eye(self.weights.shape[1])[predictions[j]])
                    biases_grad += np.eye(self.bias.shape[0])[batch_targets[j]] - np.eye(self.bias.shape[0])[
                        predictions[j]]
                self.weights += self.learning_rate * gradients / batch_inputs.shape[0]
                self.bias += self.learning_rate * biases_grad / batch_inputs.shape[0]

    def train_sgd(self, inputs, targets, num_epochs):
        for epoch in range(num_epochs):
            for i in np.random.permutation(inputs.shape[0]):
                y = self.predict(inputs[i])
                if y != targets[i]:
                    update = np.transpose(self.learning_rate * (np.expand_dims(inputs[i], axis=1) @ np.expand_dims(
                        np.eye(self.weights.shape[1])[targets[i]], axis=0) - np.expand_dims(inputs[i],
                                                                                            axis=1) @ np.expand_dims(
                        np.eye(self.weights.shape[1])[y], axis=0)))
                    self.weights += update.T
                    self.bias[targets[i]] -= self.learning_rate
                    self.bias[y] += self.learning_rate

    def plot_data(self, inputs, targets):
        colors = ['r', 'g', 'b']
        markers = ['o', '^', 's']
        for i in range(targets.max() + 1):
            plt.scatter(inputs[:, 0][targets == i], inputs[:, 1][targets == i], color=colors[i], marker=markers[i])
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.show()

    def plot_boundary(self, inputs, targets):
        x_min, x_max = inputs[:, 0].min() - 1, inputs[:, 0].max() + 1
        y_min, y_max = inputs[:, 1].min() - 1, inputs[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))

        if self.topology == 'single':
            Z = np.array([self.predict([x1, x2]) for x1, x2 in zip(xx.ravel(), yy.ravel())])
        elif self.topology == 'multi':
            Z = np.array([np.argmax(self.predict_multi([x1, x2])) for x1, x2 in zip(xx.ravel(), yy.ravel())])

        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.4)
        self.plot_data(inputs, targets)