import math

import matplotlib.pyplot as plt
import numpy as np


# upravit strukturu, pochopit
def knn(X_train, y_train, X_test, k=1):
    y_pred = []
    for i in range(len(X_test)):
        distances = []
        for j in range(len(X_train)):
            distance = math.sqrt(sum([(X_test[i][k] - X_train[j][k]) ** 2 for k in range(len(X_test[i]))]))
            distances.append((distance, y_train[j]))
        distances.sort()
        k_nearest_neighbors = distances[:k]
        k_nearest_labels = [label for (dist, label) in k_nearest_neighbors]
        most_common_label = max(set(k_nearest_labels), key=k_nearest_labels.count)
        y_pred.append(most_common_label)
    return y_pred


def knn_plot(X_train, y_train, X_test, k=1):
    y_pred = knn(X_train, y_train, X_test, k)
    # Determination of range of space
    min_values = np.min(X_train, axis=0)
    max_values = np.max(X_train, axis=0)

    # Creation of meshgrid
    x_values, y_values = np.meshgrid(np.linspace(min_values[0], max_values[0], 100),
                                     np.linspace(min_values[1], max_values[1], 100))
    meshgrid = np.vstack([x_values.ravel(), y_values.ravel()]).T

    meshgrid_codes = knn(X_train, y_train, meshgrid, k)
    meshgrid_codes = np.array(meshgrid_codes).reshape(x_values.shape)

    plt.figure(figsize=(8, 8))
    plt.contourf(x_values, y_values, meshgrid_codes, alpha=0.2, levels=np.arange(3 + 1) - 0.5, cmap='jet')

    # plot data points
    colors = ['red', 'blue', 'green', 'black', 'purple']
    for i, color in zip(np.unique(y_pred), colors):
        idx = np.where(np.array(y_pred) == i)
        plt.scatter(np.array(X_train)[idx, 0], np.array(X_train)[idx, 1], color=color, label=f"Shluk {i + 1}")

    plt.xlabel('x')
    plt.ylabel('y')
    # plt.title("k-NN")
    plt.legend(loc='best')
    plt.savefig("./pics/knn_k=2.eps", format='eps', dpi=300)
    plt.show()
