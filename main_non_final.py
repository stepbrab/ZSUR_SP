import matplotlib.pyplot as plt
import numpy as np

from perceptron import Perceptron

from divisive import div_plot_get_clusters

from maximin import mm_plot_and_get_clusters

from agglomerative import dist_matrix, agg_plot_and_get_clusters

from kmeans import plot_kmeans, plot_bin_split

from perceptron import Perceptron

from iterative_opt import it_opt

from bayes import bayes

from vector_quantization import vq_plot

from k_nearest_neighbour import knn_plot

from lin_disc_func import  plot_rosenblatt, plot_const_incr


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
    plt.savefig("./pics/data.eps", format='eps', dpi=300)
    plt.show()


if __name__ == "__main__":
    # Načtení dat
    data = load("data.txt")

    data = data[::10]  # Zmenšení obsahu dat pro rychlejší výpočty

    dm = dist_matrix(data)

    # # Vykreslení původních dat
    # plot_data(data)

    # # Výpočet počtů shluků a ploty výsledků
    # amount_of_classes_agg = agg_plot_and_get_clusters(data)
    # print("Počet shluků odhadnut metodou shlukové hladiny: ", amount_of_classes_agg)
    #
    # cutoff_dist = np.mean(dm) * len(dm) / 800 #kolik zvolit?
    # amount_of_classes_div = div_plot_get_clusters(data, 0, cutoff_dist)
    # print("Počet shluků odhadnut metodou řetězové mapy: ", amount_of_classes_div)
    #
    # q = 0.9
    # amount_of_classes_mm = mm_plot_and_get_clusters(data, q)
    # print("Počet shluků odhadnut metodou maximin: ", amount_of_classes_mm)
    #
    # # Test shodnosti výsledků všech metod
    # if amount_of_classes_agg == amount_of_classes_div == amount_of_classes_mm:
    #     print("Počty shluků všech metod se shodují. Počet shluků je:", amount_of_classes_agg)
    # else:
    #     print(
    #         "Počty shluků všech metod se liší.")

    # K-means
    clusters, labels = plot_kmeans(data, amount_of_classes=3)
    clusters_bin, labels_bin = plot_bin_split(data, amount_of_classes=3) #?

    # # Iterativní optimalizace
    # it_opt(clusters)
    # it_opt(clusters_bin)

    # # Bayesův klasifikátor
    # bayes(data, clusters)

    # # Vektorová kvantizace
    # vq_plot(data, clusters)

    # # Klasifikátor podle nejbližšího souseda
    # knn_plot(data, labels, data, k=1)
    # knn_plot(data, labels, data, k=2)

    # Klasifikátor s lineárními diskriminačními funkcemi
    # plot_rosenblatt(data, labels, 10)
    # plot_const_incr(data, labels, 10, 0.1)

    # # Neuronová síť
    # # Inicializace
    # perceptron_batch_gd = Perceptron(num_inputs=2, num_outputs=3, learning_rate=0.1)
    # perceptron_sgd = Perceptron(num_inputs=2, num_outputs=3, learning_rate=0.1)
    # perceptron_batch_gd_multi = Perceptron(num_inputs=2, num_outputs=3, learning_rate=0.1, topology='multi')
    # perceptron_sgd_multi = Perceptron(num_inputs=2, num_outputs=3, learning_rate=0.1, topology='multi')
    #
    # # Trénování
    # perceptron_batch_gd.train_batch_gd(data, labels, num_epochs=100, batch_size=None)
    # perceptron_sgd.train_sgd(data, labels, num_epochs=100)
    # perceptron_batch_gd_multi.train_batch_gd(data, labels, num_epochs=100, batch_size=None)
    # perceptron_sgd_multi.train_sgd(data, labels, num_epochs=100)
    #
    # # Plot
    # perceptron_batch_gd.plot_boundary(data, labels)
    # perceptron_sgd.plot_boundary(data, labels)
    # perceptron_batch_gd_multi.plot_boundary(data, labels)
    # perceptron_sgd_multi.plot_boundary(data, labels)
