import matplotlib.pyplot as plt

from agglomerative import agg

from iterative_opt import it_opt

import numpy as np

import divisive
import k_nearest_neighbour

from kmeans import kmeans, kmeans_bin
from lin_disc_func import rosenblatt, constant_increment, plot_ros_and_const_incr

import maximin

import bayes
import vector_quantization


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


def plot_clusters(clusters):
    for point in clusters[0]:
        plt.plot(point[0], point[1], marker=".", color='r')
    for point in clusters[1]:
        plt.plot(point[0], point[1], marker=".", color='g')
    for point in clusters[2]:
        plt.plot(point[0], point[1], marker=".", color='b')
    plt.show()


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

    w_rosenblatt = rosenblatt(data, labels)
    w_constant_increment = constant_increment(data, labels)

    plot_ros_and_const_incr(data, labels, w_rosenblatt, w_constant_increment)


    ##decision boundary se to jmenuje ty demente




