import numpy as np
import matplotlib.pyplot as plt


def rosenblatt_classifier(X, y, learning_rate=0.1, max_epochs=1000, bias=True):
    n_samples, n_features = X.shape

    # Initialize weights
    weights = np.zeros(n_features + int(bias))

    # Add bias term if necessary
    if bias:
        X = np.concatenate((X, np.ones((n_samples, 1))), axis=1)

    # Train the model
    for epoch in range(max_epochs):
        for i in range(n_samples):
            activation = np.dot(X[i], weights)
            if activation >= 0.0:
                y_pred = 1
            else:
                y_pred = -1
            error = y[i] - y_pred
            weights += learning_rate * error * X[i]

    # Define the predict function
    def predict(X_new):
        if bias:
            X_new = np.concatenate((X_new, np.ones((X_new.shape[0], 1))), axis=1)
        return np.where(np.dot(X_new, weights) >= 0.0, 1, -1)

        # Plot the data and the decision boundary
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
        Z = predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.figure(figsize=(8, 8))
        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolors='k')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Rosenblatt Classifier')
        plt.show()

    return weights
