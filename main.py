import matplotlib.pyplot as plt
import numpy as np

from kmeans import kmeans
from neuralnetwork import Perceptron

from divisive import div_plot_get_clusters

from maximin import mm_plot_and_get_clusters

from agglomerative import dist_matrix, agg_plot_and_get_clusters

from kmeans import kmeans, kmeans_bin

from neuralnetwork import Perceptron

from iterative_opt import it_opt

from bayes import bayes

from vector_quantization import vq_plot

from k_nearest_neighbour import knn_plot

from lin_disc_func import plot_ros_and_const_incr, compare_iterations, rosenblatt, constant_increment

def load(infile):
    # Zkontrolovat
    # Načte vektory ze souboru do pole 2D vektorů
    file = open(infile, "rt")
    data = []
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
        data.append(vector)
    file.close()
    data = np.array(data)
    return data


if __name__ == "__main__":
    # Načtení dat
    data = load("data.txt")

    data = data[::10]  # Zmenšení obsahu dat pro rychlejší výpočty

    dm = dist_matrix(data)

    # Výpočet počtů shluků a ploty výsledků
    amount_of_classes_agg = agg_plot_and_get_clusters(data)
    print("Počet shluků odhadnut metodou shlukové hladiny: ", amount_of_classes_agg)

    cutoff_dist = np.mean(dm) * len(dm) / 1200
    amount_of_classes_div = div_plot_get_clusters(data, 0, cutoff_dist)
    print("Počet shluků odhadnut metodou řetězové mapy: ", amount_of_classes_div)

    cutoff = np.mean(dm) * 1.5
    amount_of_classes_mm = mm_plot_and_get_clusters(data, cutoff)
    print("Počet shluků odhadnut metodou maximin: ", amount_of_classes_mm)

    # Test shodnosti výsledků všech metod
    if amount_of_classes_agg == amount_of_classes_div == amount_of_classes_mm:
        print("Počty shluků všech metod se shodují. Počet shluků je:", amount_of_classes_agg)
    else:
        print(
            "Počet shluků všech metod se liší. Je třeba zkusit jiný cutoff_dist, či změnit počáteční bod metody "
            "řetězové mapy.")

    #K-means
    clusters, labels = kmeans(data, amount_of_classes=3)
    clusters_bin, labels_bin = kmeans_bin(data, amount_of_classes=3)

    #Iterativní optimalizace
    it_opt(clusters)
    it_opt(clusters_bin)

    #Bayesův klasifikátor
    bayes(data, clusters)

    #Vektorová kvantizace s 10 iteracemi
    vq_plot(data, 10, clusters)

    #Klasifikátor podle nejbližšího souseda
    knn_plot(data, labels, data, k=1)
    knn_plot(data, labels, data, k=2)

    #Klasifikátor s lineárními diskriminačními funkcemi
    w_rosenblatt = rosenblatt(data, labels)
    w_constant_increment = constant_increment(data, labels)
    plot_ros_and_const_incr(data, labels, w_rosenblatt, w_constant_increment)
    compare_iterations(data, labels)

    #Neuronová síť
    #Inicializace
    perceptron_batch_gd = Perceptron(num_inputs=2, num_outputs=3, learning_rate=0.1)
    perceptron_sgd = Perceptron(num_inputs=2, num_outputs=3, learning_rate=0.1)
    perceptron_batch_gd_multi = Perceptron(num_inputs=2, num_outputs=3, learning_rate=0.1, topology='multi')
    perceptron_sgd_multi = Perceptron(num_inputs=2, num_outputs=3, learning_rate=0.1, topology='multi')

    #Trénování
    perceptron_batch_gd.train_batch_gd(data, labels, num_epochs=100, batch_size=None)
    perceptron_sgd.train_sgd(data, labels, num_epochs=100)
    perceptron_batch_gd_multi.train_batch_gd(data, labels, num_epochs=100, batch_size=None)
    perceptron_sgd_multi.train_sgd(data, labels, num_epochs=100)

    #Plot
    perceptron_batch_gd.plot_boundary(data, labels)
    perceptron_sgd.plot_boundary(data, labels)
    perceptron_batch_gd_multi.plot_boundary(data, labels)
    perceptron_sgd_multi.plot_boundary(data, labels)
