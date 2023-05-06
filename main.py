import matplotlib.pyplot as plt
import numpy as np

from kmeans import kmeans
from perceptron import Perceptron


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


def dist_matrix(data):
    # Zkontrolovat
    # Výpočet matice vzdáleností
    dm = np.sqrt(np.sum((data[:, np.newaxis, :] - data[np.newaxis, :, :]) ** 2, axis=2))
    return dm


if __name__ == "__main__":
    data = load("data.txt")

    data = data[::10]

    dm = dist_matrix(data)

    # amount_of_classes_agg = agg(data)
    # print("Amount of classes estimated by the agglomerative method: ", amount_of_classes_agg)

    # amount_of_classes_div = divisive.div(data, 150)
    # print("Amount of classes estimated by the divisive chain map method: ", amount_of_classes_div)

    # cutoff = np.mean(dm) * 1.5
    # amount_of_classes_mm = maximin.mm(data, cutoff)
    # print("Amount of classes estimated by the maximin method: ", amount_of_classes_mm)

    # if amount_of_classes_agg == amount_of_classes_div == amount_of_classes_mm:
    #     kmeans = kmeans.kmeans(data, amount_of_classes_agg)
    # else:
    #     print("Try a different cutoff distance.")

    clusters, labels = kmeans(data, amount_of_classes=3)
    # clusters, labels = kmeans_bin(data, amount_of_classes=3)

    # bayes.bayesian_classifier(data, clusters)

    # Vektorová kvantizace s kódovou knihou o velikosti 5 a 10 iteracemi
    # vector_quantization.vq_plot(data, 10, clusters)

    # k_nearest_neighbour.knn_plot(data, labels, data, 1)

    # w_rosenblatt = rosenblatt(data, labels)
    # w_constant_increment = constant_increment(data, labels)
    #
    # plot_ros_and_const_incr(data, labels, w_rosenblatt, w_constant_increment)

    # Create the perceptron object
    perceptron_batch_gd = Perceptron(num_inputs=2, num_outputs=3, learning_rate=0.1)
    perceptron_sgd = Perceptron(num_inputs=2, num_outputs=3, learning_rate=0.1)
    perceptron_batch_gd_multi = Perceptron(num_inputs=2, num_outputs=3, learning_rate=0.1, topology='multi')
    perceptron_sgd_multi = Perceptron(num_inputs=2, num_outputs=3, learning_rate=0.1, topology='multi')

    # Train the perceptron using Batch Gradient Descent
    perceptron_batch_gd.train_batch_gd(data, labels, num_epochs=100, batch_size=None)
    perceptron_sgd.train_sgd(data, labels, num_epochs=100)
    perceptron_batch_gd_multi.train_batch_gd(data, labels, num_epochs=100, batch_size=None)
    perceptron_sgd_multi.train_sgd(data, labels, num_epochs=100)
    # Plot the decision boundary
    perceptron_batch_gd.plot_boundary(data, labels)
    perceptron_sgd.plot_boundary(data, labels)
    perceptron_batch_gd.plot_boundary(data, labels)
    perceptron_sgd.plot_boundary(data, labels)
