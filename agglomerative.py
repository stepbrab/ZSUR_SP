import matplotlib.pyplot as plt
import numpy as np


def dist_matrix(data):
    dm = np.sqrt(np.sum((data[:, np.newaxis, :] - data[np.newaxis, :, :]) ** 2, axis=2))
    return dm


def agg_plot_and_get_clusters(data):
    dm = dist_matrix(data)

    np.fill_diagonal(dm, np.inf)

    dists = np.zeros(dm.shape[0] - 1)

    n = 0

    while dm.shape[0] > 1:
        print('Probíhá metoda shlukové hladiny [' + str(n) + '/' + str(len(data)) + '].')
        x, y = np.unravel_index(np.argmin(dm), dm.shape)
        min_dist = dm[x, y]
        dm[x, :] = np.minimum(dm[x, :], dm[y, :])
        dm[:, x] = np.minimum(dm[:, x], dm[:, y])
        np.fill_diagonal(dm, np.inf)
        dm = np.delete(dm, y, axis=0)
        dm = np.delete(dm, y, axis=1)
        dists[n] = min_dist
        n += 1

    max_dist_diff = 0
    max_dist_diff_i = 0
    for i in range(1, len(dists)):
        if max_dist_diff < np.abs(dists[i] - dists[i - 1]):
            max_dist_diff = (dists[i] - dists[i - 1])
            max_dist_diff_i = i

    amount_of_clusters = len(dists) - max_dist_diff_i + 1

    x = np.arange(1, len(data))

    plt.figure(figsize=(8, 8))
    plt.title('Metoda shlukové hladiny')
    plt.plot(x[::-1], dists[:])
    plt.axvline(x=amount_of_clusters, color='k', linestyle='-.', label='Počet shluků = ' + str(amount_of_clusters))
    plt.axhline(y=dists[-amount_of_clusters], color='r', linestyle='--',
                label='Shluková hladina h = ' + str(dists[-amount_of_clusters]))
    plt.yticks(np.arange(0, np.max(dists)))
    plt.xlim(1, 50)
    plt.gca().invert_xaxis()
    plt.legend(loc='upper left')
    plt.xlabel('Počet shluků')
    plt.ylabel('Vzdálenost mezi shluky')
    # plt.savefig("./pics/agg.eps", format='eps', dpi=300)
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.title('Metoda shlukové hladiny - zoom')
    plt.plot(x[::-1], dists[:])
    plt.axvline(x=amount_of_clusters, color='k', linestyle='-.', label='Počet shluků = ' + str(amount_of_clusters))
    plt.axhline(y=dists[-amount_of_clusters], color='r', linestyle='--',
                label='Shluková hladina h = ' + str(dists[-amount_of_clusters]))
    plt.yticks(np.arange(0, np.max(dists)))
    plt.xlim(1, 5)
    plt.gca().invert_xaxis()
    plt.legend(loc='upper left')
    plt.xlabel('Počet shluků')
    plt.ylabel('Vzdálenost mezi shluky')
    # plt.savefig("./pics/agg_zoom.eps", format='eps', dpi=300)
    plt.show()
    print('Hotovo.')

    return amount_of_clusters
