import matplotlib.pyplot as plt

import agglomerative

import iterative_opt

import numpy as np

import divisive

import kmeans

import maximin

import bayes


def load(infile):
    # Zkontrolovat
    # Načte vektory ze souboru do pole 2D vektorů
    file = open(infile, "rt")
    data = []
    for line in file:
        vector = line.split()
        for i in range(0, len(vector)):
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
    dm = dist_matrix(data)

    data = data[::10]
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
    means, variances = bayes.get_params(clusters, data)

    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    # Výpočet pravděpodobnosti pro každý bod v mřížce
    predictions = np.zeros(xx.shape)
    for i, xi in enumerate(xx):
        for j, yj in enumerate(yy):
            predictions[i, j] = bayes.classify([xi[j], yj[j]], means, variances)

    # Vykreslení výsledků
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, predictions, alpha=0.5)
    plt.scatter(data[:, 0], data[:, 1], c=clusters, alpha=0.5)
    plt.title('Bayesův klasifikátor')
    plt.xlabel('První dimenze')
    plt.ylabel('Druhá dimenze')
    plt.show()

    lmao