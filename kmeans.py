import numpy as np

from matplotlib import pyplot as plt


def kmeans(data, amount_of_classes):
    centers = data[np.random.choice(len(data), amount_of_classes, replace=False)]
    print('Probíhá k-means...')
    while True:
        distances = np.sqrt(((data - centers[:, np.newaxis]) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        new_centers = np.array([data[labels == i].mean(axis=0) for i in range(amount_of_classes)])
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    clusters = [data[labels == i] for i in range(amount_of_classes)]
    return clusters, labels


def plot_kmeans(data, amount_of_classes):
    clusters, labels = kmeans(data, amount_of_classes)
    plt.figure(figsize=(8, 8))
    for i, cluster in enumerate(clusters):
        plt.scatter(cluster[:, 0], cluster[:, 1], label=f"Shluk {i + 1}")
    plt.title("K-means")
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.savefig("./pics/kmeans.eps", format='eps', dpi=300)
    plt.show()
    return clusters, labels


def plot_bin_split(data, amount_of_classes):
    clusters, labels = bin_split(data, amount_of_classes)
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


def bin_split(data, amount_of_classes):
    print('Probíhá nerovnoměrné binární dělení...')
    clusters = []
    temp_data = np.array(data)
    prev_indices = np.arange(len(data))
    while True:
        curr_clusters, curr_labels = kmeans(temp_data, 2)
        for label in range(2):
            temp_data_label = temp_data[curr_labels == label]
            clusters.append((temp_data_label, np.mean(temp_data_label, axis=0),
                             np.sum(np.linalg.norm(temp_data_label - np.mean(temp_data_label, axis=0), axis=1)),
                             prev_indices[curr_labels == label]))
        if len(clusters) == amount_of_classes:
            break
        costs = np.array([cluster[2] for cluster in clusters])
        temp_data = clusters.pop(np.argmax(costs))
        prev_indices = temp_data[3]
        temp_data = temp_data[0]
    labels = np.zeros(len(data), dtype=int)
    for i, cluster in enumerate(clusters):
        labels[cluster[3]] = i
    result_clusters = [data[labels == i] for i in range(amount_of_classes)]
    return result_clusters, labels

