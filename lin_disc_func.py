import numpy as np
import random
import copy
from matplotlib import pyplot as plt


def rosenblatt(clusters, epochs=10):
    amount_of_clusters = len(clusters)
    lin_disc_funcns = [[] for i in range(amount_of_clusters)]
    for i in range(amount_of_clusters):
        cluster1 = clusters[i]
        if i == 0:
            j = 1
        else:
            j = 0
        cluster2 = clusters[j]
        for k in range(amount_of_clusters):
            if k == i or k == j:
                continue
            cluster2 = np.concatenate((cluster2, clusters[k]))
        len_c1 = len(cluster1)
        len_c2 = len(cluster2)
        datasetlabels = np.ones(len_c1 + len_c2, dtype=int)
        datasetlabels[len_c1:len_c1 + len_c2] = -1
        dataset = [np.concatenate((cluster1, cluster2), axis=0), datasetlabels]
        tempq = train_rosenblatt(dataset, epochs)
        lin_disc_funcns[i].append(tempq)
    return lin_disc_funcns


def train_rosenblatt(dataset, epochs):
    len_dataset = len(dataset[0])
    mixindexes = list(range(len_dataset))
    q = np.zeros(len(dataset[0][0]) + 1) + 1
    lastdataset = copy.deepcopy(dataset)
    for epoch in range(epochs):
        random.shuffle(mixindexes)
        for index in range(len_dataset):
            lastdataset[0][index, :] = dataset[0][mixindexes[index]]
            lastdataset[1][index] = dataset[1][mixindexes[index]]
        for i in range(len_dataset):
            tempbod = lastdataset[0][i]
            tempbod = np.insert(tempbod, 0, 1)
            templabel = lastdataset[1][i]
            if q.T.dot(tempbod) >= 0:
                w = 1
            else:
                w = -1
            if w == templabel:
                continue
            else:
                q = q.T + tempbod.T.dot(templabel)
    return q


def classify_rosenblatt(data, q):
    datalabels = np.zeros(len(data), dtype=int) + len(q)
    q = np.asarray(q)
    for i in range(len(data)):
        bod = data[i]
        bod = np.insert(bod, 0, 1)
        label = len(q)
        rozhodnuti = []
        for j in range(len(q)):

            temp = q[j][0].T.dot(bod)
            if temp >= 0:
                rozhodnuti.append(True)
                label = j
            else:
                continue
        if len(rozhodnuti) > 1 or len(rozhodnuti) == 0:
            label = len(q)
        datalabels[i] = label
    return datalabels


def plot_rosenblatt(data, clusters, epochs=10):
    q = rosenblatt(clusters, epochs)
    xmin, xmax = np.min(data[:, 0]), np.max(data[:, 0])
    ymin, ymax = np.min(data[:, 1]), np.max(data[:, 1])
    x = np.linspace(xmin, xmax, 2)
    y = []
    q = np.asarray(q)
    print(q)
    for j in range(1):
        for i in range(3):
            y.append((-q[i][j][0] - q[i][j][1] * x) / q[i][j][2])
    plt.figure(figsize=(8, 8))
    for i in range(3):
        plt.plot(x, y[i])

    min_values = np.min(data, axis=0)
    max_values = np.max(data, axis=0)
    x_values, y_values = np.meshgrid(np.linspace(min_values[0], max_values[0], 100),
                                     np.linspace(min_values[1], max_values[1], 100))
    meshgrid_points = np.vstack([x_values.ravel(), y_values.ravel()]).T
    meshgrid_labels = classify_rosenblatt(meshgrid_points, q)
    meshgrid_labels = meshgrid_labels.reshape(x_values.shape)
    plt.contourf(x_values, y_values, meshgrid_labels, alpha=1, cmap='ocean')
    for i, cluster in enumerate(clusters):
        plt.scatter(cluster[:, 0], cluster[:, 1], label=f"Shluk {i + 1}")
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.title('Rosenblatt - rastr')
    # plt.savefig("./pics/rosenblatt_rastr.eps", format='eps', dpi=300)
    plt.show()

    plt.figure(figsize=(8, 8))
    for i, cluster in enumerate(clusters):
        plt.scatter(cluster[:, 0], cluster[:, 1], label=f"Shluk {i + 1}")
    for j in range(1):
        for i in range(3):
            y.append((-q[i][j][0] - q[i][j][1] * x) / q[i][j][2])
    for i in range(3):
        plt.plot(x, y[i])
    plt.title('Rosenblatt - lin. diskr. fce.')
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    # plt.savefig("./pics/rosenblatt_ldf.eps", format='eps', dpi=300)
    plt.show()


def const_incr(clusters, epochs=10, beta=0.1):
    amount_of_classes = len(clusters)
    linDiskrFcns = [[] for i in range(amount_of_classes)]
    for i in range(amount_of_classes):
        cluster1 = clusters[i]
        if i == 0:
            j = 1
        else:
            j = 0
        cluster2 = clusters[j]
        for k in range(amount_of_classes):
            if k == i or k == j:
                continue
            cluster2 = np.concatenate((cluster2, clusters[k]))

        len_c1 = len(cluster1)
        len_c2 = len(cluster2)
        datasetlabels = np.ones(len_c1 + len_c2, dtype=int)
        datasetlabels[len_c1:len_c1 + len_c2] = -1
        dataset = [np.concatenate((cluster1, cluster2), axis=0), datasetlabels]
        tempq = train_const_incr(dataset, epochs, beta)
        linDiskrFcns[i].append(tempq)
    return linDiskrFcns


def train_const_incr(dataset, epochs, beta):
    pocetDat = len(dataset[0])
    mixindexes = list(range(pocetDat))
    q = np.zeros(len(dataset[0][0]) + 1) + 1
    lastdataset = copy.deepcopy(dataset)
    for epoch in range(epochs):
        random.shuffle(mixindexes)
        for index in range(pocetDat):
            lastdataset[0][index, :] = dataset[0][mixindexes[index]]
            lastdataset[1][index] = dataset[1][mixindexes[index]]
        for i in range(pocetDat):
            tempbod = lastdataset[0][i]
            tempbod = np.insert(tempbod, 0, 1)
            templabel = lastdataset[1][i]
            if q.T.dot(tempbod) >= 0:
                w = 1
            else:
                w = -1

            if w == templabel:
                continue
            else:
                b = abs(np.dot(q.T, tempbod) * templabel) / beta
                c = (beta * b) / np.sum(tempbod ** 2)
                q = q.T + c * tempbod.T.dot(templabel)
    return q


def classify_const_incr(data, q):
    datalabels = np.zeros(len(data), dtype=int)
    q = np.asarray(q)

    for i in range(len(data)):
        bod = data[i]
        bod = np.insert(bod, 0, 1)
        label = len(q)
        rozhodnuti = []
        for j in range(len(q)):

            temp = q[j][0].T.dot(bod)
            if temp >= 0:
                rozhodnuti.append(True)
                label = j
            else:
                continue
        if len(rozhodnuti) > 1 or len(rozhodnuti) == 0:
            label = len(q)
        datalabels[i] = label
    return datalabels


def plot_const_incr(data, clusters, epochs=10, beta=0.1):
    q = const_incr(clusters, epochs, beta)
    xmin, xmax = np.min(data[:, 0]), np.max(data[:, 0])
    ymin, ymax = np.min(data[:, 1]), np.max(data[:, 1])
    x = np.linspace(xmin, xmax, 2)
    y = []
    q = np.asarray(q)
    print(q)
    for j in range(1):
        for i in range(3):
            y.append((-q[i][j][0] - q[i][j][1] * x) / q[i][j][2])
    plt.figure(figsize=(8, 8))
    for i in range(3):
        plt.plot(x, y[i])

    min_values = np.min(data, axis=0)
    max_values = np.max(data, axis=0)
    x_values, y_values = np.meshgrid(np.linspace(min_values[0], max_values[0], 100),
                                     np.linspace(min_values[1], max_values[1], 100))
    meshgrid_points = np.vstack([x_values.ravel(), y_values.ravel()]).T
    meshgrid_labels = classify_const_incr(meshgrid_points, q)
    meshgrid_labels = meshgrid_labels.reshape(x_values.shape)
    plt.contourf(x_values, y_values, meshgrid_labels, alpha=1, cmap='ocean')
    for i, cluster in enumerate(clusters):
        plt.scatter(cluster[:, 0], cluster[:, 1], label=f"Shluk {i + 1}")
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.title('Const incr - rastr')
    # plt.savefig("./pics/const_incr_rastr.eps", format='eps', dpi=300)
    plt.show()

    plt.figure(figsize=(8, 8))
    for i, cluster in enumerate(clusters):
        plt.scatter(cluster[:, 0], cluster[:, 1], label=f"Shluk {i + 1}")
    for j in range(1):
        for i in range(3):
            y.append((-q[i][j][0] - q[i][j][1] * x) / q[i][j][2])
    for i in range(3):
        plt.plot(x, y[i])
    plt.title('Const incr - lin. diskr. fce.')
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    # plt.savefig("./pics/const_incr_ldf.eps", format='eps', dpi=300)
    plt.show()
