import numpy as np
from matplotlib import pyplot as plt


# V poradku az na vykresleni, taky nevim co ty stredy.. jestli je to ok takhle.. checknu soubory od petra, mozna se taky vyseru na vykresleni protoze vysledek to dava dobrej.
def dist_matrix(data):
    # Zkontrolovat
    # Výpočet matice vzdáleností
    dm = np.sqrt(np.sum((data[:, np.newaxis, :] - data[np.newaxis, :, :]) ** 2, axis=2))
    return dm


def mm_plot_and_get_clusters(data, cutoff_dist):
    dm = dist_matrix(data)
    # cutoff dist pro data:10 funguje, data:5 nefunguje
    n = dm.shape[0]
    centers = [0]
    dist_to_centers = dm[0]
    while len(centers) < n:
        # najdeme nejvzdalenejsi bod od stredů shluků
        i = np.argmax(dist_to_centers)
        dist = dist_to_centers[i]
        if dist > cutoff_dist:
            centers.append(i)
            # aktualizujeme vzdálenosti pro nový střed
            new_d = dm[i]
            dist_to_centers = np.minimum(dist_to_centers, new_d)
        else:
            break

    # Rozdělení bodů do shluků
    clusters = [[] for _ in range(len(centers))]
    for i in range(data.shape[0]):
        distances = dm[i][centers]
        closest_center_index = np.argmin(distances)
        clusters[closest_center_index].append(i)

    plt.figure(figsize=(8, 8))
    # plt.title('Metoda maximin')
    for i in range(len(clusters)):
        cluster_indices = clusters[i]
        cluster_points = data[cluster_indices]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Shluk {i + 1}")

    plt.scatter(data[centers][:, 0], data[centers][:, 1], marker="x", s=100, linewidth=3, color="black",
                label="Středy")
    plt.legend()
    plt.savefig("./pics/mm.eps", format='eps', dpi=300)
    plt.show()

    amount_of_clusters = len(centers)
    return amount_of_clusters
