import numpy as np

from matplotlib import pyplot as plt


# kmeans_bin nevim, zda ok

def kmeans(data, amount_of_classes):
    centers = data[np.random.choice(len(data), amount_of_classes, replace=False)]
    cenaTrid = np.zeros(amount_of_classes)

    while True:
        distances = np.sqrt(((data - centers[:, np.newaxis]) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        new_centers = np.array([data[labels == i].mean(axis=0) for i in range(amount_of_classes)])
        if np.allclose(new_centers, centers):
            break
        centers = new_centers

    clusters = [data[labels == i] for i in range(amount_of_classes)]
    for i in range(amount_of_classes):  # Výpočet ceny jednotlivých tříd
        distances_i = np.array(np.sqrt(((clusters[i] - centers[i]) ** 2).sum(axis=1)))
        cenaTrid[i] = np.sum(distances_i)
    return clusters, labels, cenaTrid


def plot_kmeans(data, amount_of_classes):
    clusters, labels, cenaTrid = kmeans(data, amount_of_classes)
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


def plot_kmeans_bin(data, amount_of_classes):
    clusters, labels = kmeans_bin(data, amount_of_classes)
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


def kmeans_bin(data, amount_of_classes):  # ?
    tridy = []
    tempdata = np.array(data)
    prevClusters = np.array(range(len(data)))
    while True:
        clusters, labels, cenaTrid = kmeans(data, 2)
        centers = np.array([data[labels == i].mean(axis=0) for i in range(2)])


        #jsem zde
        tempdata1 = tempdata[clusters[0]]
        tridy.append((tempdata1, centers[0], cenaTrid[0], prevClusters[clusters[0]]))

        tempdata2 = tempdata[clusters[1]]
        tridy.append((tempdata2, centers[1], cenaTrid[1], prevClusters[clusters[1]]))
        if len(tridy) == amount_of_classes:
            break
        ceny = np.zeros(len(tridy), dtype=int)
        for j in range(len(tridy)):
            ceny[j] = tridy[j][2]
        tempdata = tridy.pop(np.argmax(ceny))
        prevClusters = tempdata[3]
        tempdata = tempdata[0]

        new_centers = np.array(new_centers)
        if np.allclose(new_centers, centers):
            break

    labels = np.zeros(len(data), dtype=int)
    stredy = []
    for i in range(amount_of_classes):
        labels[tridy[i][3]] = i
        stredy.append(tridy[i][1])
    clusters = [data[labels == i] for i in range(amount_of_classes)]
    return clusters, labels
