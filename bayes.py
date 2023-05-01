import numpy as np
import matplotlib.pyplot as plt

#upravit strukturu, pochopit
def normal_density(x, mean, variance):
    return np.exp(-(x - mean) ** 2 / (2 * variance)) / np.sqrt(2 * np.pi * variance)


def bayes(data, clusters):
    # Estimation of mean and variance for each class
    means = [np.mean(c, axis=0) for c in clusters]
    variances = [np.var(c, axis=0) for c in clusters]

    # Number of classes
    num_classes = len(clusters)

    # Determination of range of space
    min_values = np.min(data, axis=0)
    max_values = np.max(data, axis=0)

    # Creation of meshgrid
    x_values, y_values = np.meshgrid(np.linspace(min_values[0], max_values[0], 100),
                                     np.linspace(min_values[1], max_values[1], 100))
    meshgrid_points = np.vstack([x_values.ravel(), y_values.ravel()]).T

    # Calculation of probability for each point in meshgrid for each class
    probabilities = np.zeros((num_classes, meshgrid_points.shape[0]))
    for i in range(num_classes):
        probabilities[i] = np.prod(normal_density(meshgrid_points, means[i], variances[i]), axis=1)

    # Assigning each point in meshgrid to class with highest probability
    predicted_classes = np.argmax(probabilities, axis=0)

    # Reshape and return meshgrid and assigned classes
    meshgrid_classes = predicted_classes.reshape(x_values.shape)

    # Plot meshgrid and clusters
    plt.figure(figsize=(8, 8))
    plt.contourf(x_values, y_values, meshgrid_classes, alpha=0.2, levels=np.arange(num_classes + 1) - 0.5, cmap='jet')
    for i in range(num_classes):
        plt.scatter(clusters[i][:, 0], clusters[i][:, 1])
    plt.xlim(min_values[0], max_values[0])
    plt.ylim(min_values[1], max_values[1])
    plt.title('Bayes')
    plt.show()
    return meshgrid_classes
