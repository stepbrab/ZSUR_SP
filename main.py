import matplotlib.pyplot as plt

import agglomerative

import iterative_opt

import numpy as np

import divisive

import kmeans

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

    # dm = dist_matrix(data)

    # print(len(data))
    #
    # for point in data:  # start plot
    #     plt.plot(point[0], point[1], marker=".")
    # plt.show()
    #
    # # x = maximin.mm(data, cutoff)
    # # for point in x[1]:  # start plot
    # #     plt.plot(data[point, 0], data[point, 1], "kx", markersize=10)
    # # plt.show()
    # # print(x)
    #
    # cutoff = np.mean(dm)
    # amount_of_classes_agg = agglomerative.agg(data, cutoff)
    # print("Amount of classes estimated by the agglomerative method: ", amount_of_classes_agg)
    #
    # amount_of_classes_div = divisive.div(data, 0)
    # print("Amount of classes estimated by the divisive chain map method: ", amount_of_classes_div)
    #
    # cutoff = np.mean(dm)
    # amount_of_classes_mm = maximin.mm(data, cutoff)
    # print("Amount of classes estimated by the maximin method: ", amount_of_classes_mm)
    #
    # if amount_of_classes_agg == amount_of_classes_div == amount_of_classes_mm:
    #     kmeans = kmeans.kmeans(data, amount_of_classes_agg)
    # else:
    #     print("Try a different cutoff distance.")

    # diff = 0
    # while diff == 0:
    #     clusters = kmeans.kmeans(data, 3)
    #     clusters_opt, diff = iterative_opt.it_opt(clusters)
    #     if diff > 0:
    #         plot_clusters(clusters)
    #         plot_clusters(clusters_opt)
    #         break

    clusters = kmeans.kmeans(data, 3)

    # bayes.bayesian_classifier(data, clusters)

    # Vektorová kvantizace s kódovou knihou o velikosti 5 a 10 iteracemi
    codes, codebook = vector_quantization.vq(data, 3, 10)

    # Vykreslení trénovacích dat s barvami odpovídajícími přiřazeným kódům
    plt.scatter(data[:, 0], data[:, 1], c=codes)
    # Vykreslení kódové knihy jako červených křížků
    plt.scatter(codebook[:, 0], codebook[:, 1], marker='x', c='red')
    plt.show()



