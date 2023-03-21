import math
import numpy as np
import matplotlib.pyplot as plt


def knn_classifier(X_train, y_train, X_test, k=3):
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

    # create meshgrid
    h = .02  # step size in the mesh
    x_min, x_max = np.array(X_train)[:, 0].min() - 1, np.array(X_train)[:, 0].max() + 1
    y_min, y_max = np.array(X_train)[:, 1].min() - 1, np.array(X_train)[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # predict on meshgrid points
    Z = np.array(knn_classifier(X_train, y_train, np.c_[xx.ravel(), yy.ravel()], k)).reshape(xx.shape)

    # plot the meshgrid and data points
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    # plot data points
    colors = ['red', 'blue', 'green', 'black', 'purple']
    for i, color in zip(np.unique(y_train), colors):
        idx = np.where(np.array(y_train) == i)
        plt.scatter(np.array(X_train)[idx, 0], np.array(X_train)[idx, 1], color=color, label=i)

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("k-NN classifier")
    plt.legend(loc='best')
    plt.show()

    return y_pred