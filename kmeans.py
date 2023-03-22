import numpy as np

from matplotlib import pyplot as plt


def kmeans(data, amount_of_classes):
    # inicializace centroidů
    centroids = data[np.random.choice(len(data), amount_of_classes, replace=False)]

    while True:
        # vypočítání vzdálenosti mezi všemi body a centroidy
        distances = np.sqrt(((data - centroids[:, np.newaxis]) ** 2).sum(axis=2))

        # přiřazení každého bodu k nejbližšímu centroidu
        labels = np.argmin(distances, axis=0)

        # výpočet nových centroidů
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(amount_of_classes)])

        # pokud se nové centroidy shodují s těmi předchozími, ukončit iteraci
        if np.allclose(new_centroids, centroids):
            break

        centroids = new_centroids
    clusters = [data[labels == i] for i in range(amount_of_classes)]
    return clusters, labels


def kmeans_bin(data, amount_of_classes, num_splits=10):
    # inicializace centroidů
    centroids = data[np.random.choice(len(data), amount_of_classes, replace=False)]
    while True:
        # vypočítání vzdálenosti mezi všemi body a centroidy
        distances = np.sqrt(((data - centroids[:, np.newaxis]) ** 2).sum(axis=2))

        # přiřazení každého bodu k nejbližšímu centroidu
        labels = np.argmin(distances, axis=0)

        # výpočet nových centroidů
        new_centroids = []
        for i in range(amount_of_classes):
            # vybere body příslušící k i-tému clusteru
            cluster_points = data[labels == i]

            # rozdělení clusteru na podmnožiny
            num_points = len(cluster_points)
            split_points = np.linspace(0, num_points, num_splits + 1, dtype=np.int)[1:-1]
            sub_clusters = np.split(cluster_points, split_points)

            # nalezení centroidu pro každou podmnožinu
            sub_centroids = [sub_cluster.mean(axis=0) for sub_cluster in sub_clusters]

            # výpočet váženého průměru podcentroidů pro i-tý cluster
            sub_cluster_sizes = [len(sub_cluster) for sub_cluster in sub_clusters]
            sub_cluster_weights = np.array(sub_cluster_sizes) / float(num_points)
            centroid_i = np.average(sub_centroids, axis=0, weights=sub_cluster_weights)
            new_centroids.append(centroid_i)

        new_centroids = np.array(new_centroids)

        # pokud se nové centroidy shodují s těmi předchozími, ukončit iteraci
        if np.allclose(new_centroids, centroids):
            break

        centroids = new_centroids
    clusters = [data[labels == i] for i in range(amount_of_classes)]
    return clusters
