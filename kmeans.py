import numpy as np

from matplotlib import pyplot as plt


# kmeans_bin nevim, zda ok

def kmeans(data, amount_of_classes):
    centers = data[np.random.choice(len(data), amount_of_classes, replace=False)]

    while True:
        distances = np.sqrt(((data - centers[:, np.newaxis]) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        new_centers = np.array([data[labels == i].mean(axis=0) for i in range(amount_of_classes)])
        if np.allclose(new_centers, centers):
            break

        centers = new_centers
    clusters = [data[labels == i] for i in range(amount_of_classes)]

    plt.figure(figsize=(8, 8))
    i = 0
    for cluster in clusters:
        plt.scatter(cluster[:, 0], cluster[:, 1], label=f"Shluk {i + 1}")
        i += 1
    plt.title("K-means")
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.savefig("./pics/kmeans.eps", format='eps', dpi=300)
    plt.show()

    return clusters, labels


def kmeans_bin(data, amount_of_classes):  # ?
    centers = data[np.random.choice(len(data), amount_of_classes, replace=False)]
    while True:
        distances = np.linalg.norm(data - centers[:, np.newaxis], axis=2)
        labels = np.argmin(distances, axis=0)
        new_centers = []
        for i in range(amount_of_classes):
            cluster_points = data[labels == i]
            num_points = len(cluster_points)
            sub_cluster_sizes = np.array_split(np.arange(num_points), 2)
            sub_clusters = [cluster_points[indices] for indices in sub_cluster_sizes]
            sub_centers = np.array([sub_cluster.mean(axis=0) for sub_cluster in sub_clusters])
            sub_cluster_weights = np.array([len(sub_cluster) for sub_cluster in sub_clusters]) / float(num_points)
            center_i = np.average(sub_centers, axis=0, weights=sub_cluster_weights)
            new_centers.append(center_i)

        new_centers = np.array(new_centers)
        if np.allclose(new_centers, centers):
            break

        centers = new_centers
    clusters = [data[labels == i] for i in range(amount_of_classes)]

    plt.figure(figsize=(8, 8))
    for i, cluster in enumerate(clusters):
        plt.scatter(cluster[:, 0], cluster[:, 1], label=f"Shluk {i + 1}")
    plt.title("Kmeans s nerovnoměrným binárním rozdělováním")
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.savefig("./pics/kmeans_bin.eps", format='eps', dpi=300)
    plt.show()

    return clusters, labels
