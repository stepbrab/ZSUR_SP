import matplotlib.pyplot as plt
import numpy as np

from divisive import div_plot_get_clusters

from maximin import mm_get_clusters

from agglomerative import dist_matrix, agg_plot_and_get_clusters

from kmeans import plot_kmeans, plot_bin_split

from perceptron import Perceptron

from iterative_opt import it_opt

from bayes import plot_bayes

from vector_quantization import vq_plot

from k_nearest_neighbour import knn_plot

from lin_disc_func import plot_rosenblatt, plot_const_incr


def load(infile):
    list_data = []
    with open(infile, "rt") as file:
        for line in file:
            vector = line.split()
            for i in range(len(vector)):
                const = 0
                if vector[i][0] == "-":
                    const = 1
                exp = int(10 * vector[i][10 + const]) + int(vector[i][11 + const])
                mark = vector[i][9 + const]
                vector[i] = float(vector[i][0 + const:8 + const])
                if mark == "+":
                    vector[i] = vector[i] * 10 ** exp
                elif mark == "-":
                    vector[i] = vector[i] / (10 ** exp)
                vector[i] = round(vector[i], 10)
                if const == 1:
                    vector[i] = -1 * vector[i]
            list_data.append(vector)
    return np.array(list_data)


def plot_data(data):
    plt.figure(figsize=(8, 8))
    plt.title('Data')
    plt.scatter(data[:, 0][:], data[:, 1][:])
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.savefig("./pics/data.eps", format='eps', dpi=300)
    plt.show()

def normalize_data(data):
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    normalized_data = (data - min_vals) / (max_vals - min_vals)
    return normalized_data

if __name__ == "__main__":
    # Načtení dat
    data = load("data.txt")

    # data = data[::10]  # Zmenšení objemu dat pro rychlejší výpočty

    data = normalize_data(data)

    dm = dist_matrix(data)

    # Vykreslení původních dat
    plot_data(data)

    # Výpočet počtů shluků a ploty výsledků
    amount_of_classes_agg = agg_plot_and_get_clusters(data)
    print("Počet shluků odhadnut metodou shlukové hladiny: ", amount_of_classes_agg)

    cutoff_dist = np.mean(dm) * 0.93
    if len(data) > 5000:
        index = 1000
    else:
        index = 10  # pro menší data
    amount_of_classes_div = div_plot_get_clusters(data, index, cutoff_dist)
    print("Počet shluků odhadnut metodou řetězové mapy: ", amount_of_classes_div)

    q = 0.95
    amount_of_classes_mm = mm_get_clusters(data, q)
    print("Počet shluků odhadnut metodou maximin: ", amount_of_classes_mm)

    amount_of_classes = amount_of_classes_agg

    # K-means
    clusters, labels = plot_kmeans(data,
                                   amount_of_classes=3)  # Může se stát, že klasifikace proběhne špatně, protože kmeans začíná s náhodnými středy
    clusters_bin, labels_bin = plot_bin_split(data, amount_of_classes=3)

    # Iterativní optimalizace
    clusters, diff = it_opt(clusters)
    clusters_bin, diff_bin = it_opt(clusters_bin)
    while diff or diff_bin != 0:
        # Zajistí správný výsledek kmeans, viz komentář u prvního použití kmeans
        clusters, labels = plot_kmeans(data, amount_of_classes=3)
        clusters_bin, labels_bin = plot_bin_split(data, amount_of_classes=3)
        clusters, diff = it_opt(clusters)
        clusters_bin, diff_bin = it_opt(clusters_bin)

    if diff_bin == 0:
        clusters = clusters_bin
        labels = labels_bin

    # Bayesův klasifikátor
    plot_bayes(data, clusters)

    # Vektorová kvantizace
    vq_plot(data, clusters)

    # Klasifikátor podle nejbližšího souseda
    knn_plot(data, labels, k=1)
    knn_plot(data, labels, k=2)

    # Klasifikátor s lineárními diskriminačními funkcemi
    max_epochs = 1000
    plot_rosenblatt(data, clusters, max_epochs)
    plot_const_incr(data, clusters, max_epochs)

    # Neuronová síť
    # Trénování SGD
    perceptron_sgd = Perceptron()
    perceptron_sgd.train_sgd(data, labels, num_epochs=100)
    perceptron_sgd.plot_boundary(data, labels, title='Perceptron_sgd')

    # Trénování batch_GD
    perceptron_sgd = Perceptron()
    perceptron_sgd.train_batch_gd(data, labels, num_epochs=100, batch_size=16)
    perceptron_sgd.plot_boundary(data, labels, title='Perceptron_batch_gd')

    print('Konec')
