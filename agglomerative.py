import time

import matplotlib.pyplot as plt

import numpy as np


# from main import dist_matrix

def dist_matrix(data):
    # Zkontrolovat
    # Výpočet matice vzdáleností
    dm = np.sqrt(np.sum((data[:, np.newaxis, :] - data[np.newaxis, :, :]) ** 2, axis=2))
    return dm


def agg(data, cutoff_dist):
    dm = dist_matrix(data)

    np.fill_diagonal(dm, np.inf)  # set diagonal to infinity

    dists = np.zeros(dm.shape[0] - 1)

    n = 0

    while dm.shape[0] > 1:
        x, y = np.unravel_index(np.argmin(dm), dm.shape)
        min_dist = dm[x, y]
        # print(dm.shape[0])
        # update distance matrix
        dm[x, :] = np.minimum(dm[x, :], dm[y, :])
        dm[:, x] = np.minimum(dm[:, x], dm[:, y])
        np.fill_diagonal(dm, np.inf)
        dm = np.delete(dm, y, axis=0)
        dm = np.delete(dm, y, axis=1)
        # print(x, y, min_dist)
        dists[n] = min_dist
        n += 1

    max_dist_diff = 0
    for i in range(1, len(dists)):
        if max_dist_diff < np.abs(dists[i] - dists[i - 1]):
            max_dist_diff = (dists[i] - dists[i - 1])
            max_dist_diff_i = i

    amount_of_clusters = len(dists) - max_dist_diff_i + 1

    # cutoff_dist = np.mean(dists)
    # cutoff_dist = max_dist_diff

    # amount_of_clusters = 1
    # for i in range(1, len(dists)):
    #     if cutoff_dist < np.abs(dists[i] - dists[i - 1]):
    #         cutoff_dist = dists[i] - dists[i - 1]
    #         amount_of_clusters += 1

    x = np.arange(1, len(data))
    plt.plot(x[::-1], dists[:])
    plt.axvline(x=amount_of_clusters, color='r', linestyle='--')
    plt.axhline(y=dists[-amount_of_clusters], color='r', linestyle='--',
                label='Shluková hladina h = ' + str(dists[-amount_of_clusters]))
    # plt.xticks(x)
    plt.yticks(np.arange(0, np.max(dists)))
    plt.xlim(1, 50)
    plt.gca().invert_xaxis()
    plt.legend(loc='upper left')
    plt.show()

    return amount_of_clusters
