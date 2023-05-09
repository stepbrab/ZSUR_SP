import numpy as np
from matplotlib import pyplot as plt


def dist_matrix(data):
    dm = np.sqrt(np.sum((data[:, np.newaxis, :] - data[np.newaxis, :, :]) ** 2, axis=2))
    return dm


def mm_plot_and_get_clusters(data, q):
    dm = dist_matrix(data)
    centers = [0]
    dist_to_centers = dm[0]
    while len(centers) < dm.shape[0]:
        i = np.argmax(dist_to_centers)
        dist = dist_to_centers[i]
        center_distances = dm[centers, i]
        avg_dist = np.mean(center_distances) * q
        if dist > avg_dist:
            centers.append(i)
            new_d = dm[i]
            dist_to_centers = np.minimum(dist_to_centers, new_d)
        else:
            break

    clusters = [[] for _ in range(len(centers))]
    for i in range(data.shape[0]):
        distances = dm[i][centers]
        closest_center_index = np.argmin(distances)
        clusters[closest_center_index].append(i)

    plt.figure(figsize=(8, 8))
    plt.title('Metoda maximin')
    for i in range(len(clusters)):
        cluster_indices = clusters[i]
        cluster_points = data[cluster_indices]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Shluk {i + 1}")

    plt.scatter(data[centers][:, 0], data[centers][:, 1], marker="x", s=100, linewidth=3, color="black",
                label="Středy")
    plt.legend()
    # plt.savefig("./pics/mm.eps", format='eps', dpi=300)
    plt.show()

    amount_of_clusters = len(centers)
    return amount_of_clusters
