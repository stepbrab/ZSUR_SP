import math

import matplotlib.pyplot as plt
import numpy as np


def knn(data, labels, test_data, k=1):
    print('Probíhá klasifikace podle nejbližšího/ch souseda/ů')
    y_pred = []
    for i in range(len(test_data)):
        distances = []
        for j in range(len(data)):
            distance = math.sqrt(sum([(test_data[i][k] - data[j][k]) ** 2 for k in range(len(test_data[i]))]))
            distances.append((distance, labels[j]))
        distances.sort()
        k_nearest_neighbors = distances[:k]
        k_nearest_labels = [label for (dist, label) in k_nearest_neighbors]
        most_common_label = max(set(k_nearest_labels), key=k_nearest_labels.count)
        y_pred.append(most_common_label)
    return y_pred


def knn_plot(data, labels, k=1):
    min_values = np.min(data, axis=0)
    max_values = np.max(data, axis=0)

    x_values, y_values = np.meshgrid(np.linspace(min_values[0], max_values[0], 100),
                                     np.linspace(min_values[1], max_values[1], 100))
    meshgrid = np.vstack([x_values.ravel(), y_values.ravel()]).T

    meshgrid_codes = knn(data, labels, meshgrid, k)
    meshgrid_codes = np.array(meshgrid_codes).reshape(x_values.shape)

    plt.figure(figsize=(8, 8))
    plt.contourf(x_values, y_values, meshgrid_codes, alpha=0.2, levels=np.arange(3 + 1) - 0.5, cmap='jet')

    colors = ['red', 'blue', 'green', 'black', 'purple']
    for i, color in zip(np.unique(labels), colors):
        idx = np.where(np.array(labels) == i)
        plt.scatter(np.array(data)[idx, 0], np.array(data)[idx, 1], color=color, label=f"Shluk {i + 1}")

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Klasifikátor podle nejbližšího souseda, počet sousedů k = " + str(k))
    plt.legend(loc='best')
    # plt.savefig("./pics/knn_k=2.eps", format='eps', dpi=300)
    plt.show()
