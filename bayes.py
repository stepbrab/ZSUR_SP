import matplotlib.pyplot as plt
import numpy as np


def normal_density(x, mean, variance):
    return np.exp(-(x - mean) ** 2 / (2 * variance)) / np.sqrt(2 * np.pi * variance)


def classify_bayes(data, means, variances, apriors):
    print('Probíhá klasifikace...')
    num_classes = len(means)
    probabilities = np.zeros((num_classes, data.shape[0]))
    for i in range(num_classes):
        probabilities[i] = apriors[i] * np.prod(normal_density(data, means[i], variances[i]), axis=1)
    predicted_classes = np.argmax(probabilities, axis=0)
    return predicted_classes


def train_bayes(clusters):
    print('Bayesův klasifikátor se trénuje...')
    len_each_cluster = sum(len(c) for c in clusters)
    print(len_each_cluster)
    apriors = [len(c) / len_each_cluster for c in clusters]  # Výpočet apriorních pravděpodobností
    means = [np.mean(c, axis=0) for c in clusters]
    variances = [np.var(c, axis=0) for c in clusters]
    return means, variances, apriors


def plot_bayes(data, clusters):
    means, variances, apriors = train_bayes(clusters)
    num_classes = len(clusters)
    min_values = np.min(data, axis=0)
    max_values = np.max(data, axis=0)
    x_values, y_values = np.meshgrid(np.linspace(min_values[0], max_values[0], 100),
                                     np.linspace(min_values[1], max_values[1], 100))
    meshgrid_points = np.vstack([x_values.ravel(), y_values.ravel()]).T
    predicted_classes = classify_bayes(meshgrid_points, means, variances, apriors)
    meshgrid_classes = predicted_classes.reshape(x_values.shape)
    plt.figure(figsize=(8, 8))
    plt.contourf(x_values, y_values, meshgrid_classes, alpha=0.2, levels=np.arange(num_classes + 1) - 0.5, cmap='ocean')
    for i in range(num_classes):
        plt.scatter(clusters[i][:, 0], clusters[i][:, 1], label=f"Shluk {i + 1}")
    plt.xlim(min_values[0], max_values[0])
    plt.ylim(min_values[1], max_values[1])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Bayesův klasifikátor')
    # plt.savefig("./pics/bayes.eps", format='eps', dpi=300)
    plt.show()
