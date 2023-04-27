import time

import matplotlib.pyplot as plt

import numpy as np

from scipy.cluster.hierarchy import dendrogram

#I bych rekl ze dobry a ready to send. Mozna dendrogram. Jinak kod good podle me.

def dist_matrix(data):
    # Zkontrolovat
    # Výpočet matice vzdáleností
    dm = np.sqrt(np.sum((data[:, np.newaxis, :] - data[np.newaxis, :, :]) ** 2, axis=2))
    return dm


def agg(data):
    dm = dist_matrix(data)

    np.fill_diagonal(dm, np.inf)  # set diagonal to infinity

    dists = np.zeros(dm.shape[0] - 1)

    n = 0

    while dm.shape[0] > 1:
        x, y = np.unravel_index(np.argmin(dm), dm.shape)
        min_dist = dm[x, y]
        # update distance matrix
        dm[x, :] = np.minimum(dm[x, :], dm[y, :])
        dm[:, x] = np.minimum(dm[:, x], dm[:, y])
        np.fill_diagonal(dm, np.inf)
        dm = np.delete(dm, y, axis=0)
        dm = np.delete(dm, y, axis=1)
        dists[n] = min_dist
        n += 1

    max_dist_diff = 0
    for i in range(1, len(dists)):
        if max_dist_diff < np.abs(dists[i] - dists[i - 1]):
            max_dist_diff = (dists[i] - dists[i - 1])
            max_dist_diff_i = i

    amount_of_clusters = len(dists) - max_dist_diff_i + 1

    x = np.arange(1, len(data))

    plt.figure(figsize=(8, 8))
    plt.plot(x[::-1], dists[:])
    plt.axvline(x=amount_of_clusters, color='k', linestyle='-.')
    plt.axhline(y=dists[-amount_of_clusters], color='r', linestyle='--',
                label='Shluková hladina h = ' + str(dists[-amount_of_clusters]))
    # plt.xticks(x)
    plt.yticks(np.arange(0, np.max(dists)))
    plt.xlim(1, 50)
    plt.gca().invert_xaxis()
    plt.legend(loc='upper left')
    plt.show()
    # plt.savefig("filepath.svg", format='svg', dpi=300)

    plt.figure(figsize=(8, 8))
    plt.plot(x[::-1], dists[:])
    plt.axvline(x=amount_of_clusters, color='k', linestyle='-.')
    plt.axhline(y=dists[-amount_of_clusters], color='r', linestyle='--',
                label='Shluková hladina h = ' + str(dists[-amount_of_clusters]))
    # plt.xticks(x)
    plt.yticks(np.arange(0, np.max(dists)))
    plt.xlim(1, 5)
    plt.gca().invert_xaxis()
    plt.legend(loc='upper left')
    plt.show()
    # plt.savefig("filepath.svg", format='svg', dpi=300)

    return amount_of_clusters
