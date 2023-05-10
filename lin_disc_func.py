import numpy as np
from matplotlib import pyplot as plt


def classify(data, q):
    data_labels = np.zeros(len(data), dtype=int) + len(q)
    q = np.asarray(q)
    for i in range(len(data)):
        point = np.insert(data[i], 0, 1)
        decisions = np.dot(q, point)
        labels = np.where(decisions >= 0)[0]
        if len(labels) == 1:
            data_labels[i] = labels[0]
    return data_labels


def rosenblatt(clusters, epochs=10):
    num_clusters = len(clusters)
    linear_discriminant_funcs = [[] for _ in range(num_clusters)]
    for i in range(num_clusters):
        cluster1 = clusters[i]
        if i == 0:
            j = 1
        else:
            j = 0
        cluster2 = clusters[j]
        for k in range(num_clusters):
            if k == i or k == j:
                continue
            cluster2 = np.concatenate((cluster2, clusters[k]))
        len_c1 = len(cluster1)
        len_c2 = len(cluster2)
        dataset_labels = np.ones(len_c1 + len_c2, dtype=int)
        dataset_labels[len_c1:len_c1 + len_c2] = -1
        dataset = [np.concatenate((cluster1, cluster2), axis=0), dataset_labels]
        temp_q = train_rosenblatt(dataset, epochs)
        linear_discriminant_funcs[i].append(temp_q)
    return linear_discriminant_funcs


def train_rosenblatt(dataset, epochs):
    len_dataset = len(dataset[0])
    q = np.zeros(len(dataset[0][0]) + 1) + 1
    for epoch in range(epochs):
        mixindexes = np.random.permutation(len_dataset)
        dataset[0] = dataset[0][mixindexes]
        dataset[1] = dataset[1][mixindexes]
        for i in range(len_dataset):
            temp_point = np.insert(dataset[0][i], 0, 1)
            temp_label = dataset[1][i]
            if q.T.dot(temp_point) >= 0:
                w = 1
            else:
                w = -1
            if w == temp_label:
                continue
            else:
                q = q.T + temp_point.T.dot(temp_label)
    return q


def plot_rosenblatt(data, clusters, epochs=10):
    print('Spouští se Rosenblattova metoda...')
    q = rosenblatt(clusters, epochs)
    xmin, xmax = np.min(data[:, 0]), np.max(data[:, 0])
    ymin, ymax = np.min(data[:, 1]), np.max(data[:, 1])
    x = np.linspace(xmin, xmax, 2)
    y = []
    q = np.asarray(q)
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
    meshgrid_labels = classify(meshgrid_points, q)
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
    num_classes = len(clusters)
    linear_discriminant_funcs = [[] for _ in range(num_classes)]

    for i in range(num_classes):
        cluster1 = clusters[i]

        if i == 0:
            j = 1
        else:
            j = 0

        cluster2 = clusters[j]

        for k in range(num_classes):
            if k == i or k == j:
                continue
            cluster2 = np.concatenate((cluster2, clusters[k]))

        len_c1 = len(cluster1)
        len_c2 = len(cluster2)

        dataset_labels = np.ones(len_c1 + len_c2, dtype=int)
        dataset_labels[len_c1:len_c1 + len_c2] = -1

        dataset = [np.concatenate((cluster1, cluster2), axis=0), dataset_labels]

        temp_q = train_const_incr(dataset, epochs, beta)
        linear_discriminant_funcs[i].append(temp_q)

    return linear_discriminant_funcs


def train_const_incr(dataset, epochs, beta):
    len_dataset = len(dataset[0])

    q = np.zeros(len(dataset[0][0]) + 1) + 1

    for epoch in range(epochs):
        mix_indexes = np.random.permutation(len_dataset)
        dataset[0] = dataset[0][mix_indexes]
        dataset[1] = dataset[1][mix_indexes]

        for i in range(len_dataset):
            temp_point = np.insert(dataset[0][i], 0, 1)
            temp_label = dataset[1][i]

            if q.T.dot(temp_point) >= 0:
                w = 1
            else:
                w = -1

            if w == temp_label:
                continue
            else:
                b = abs(np.dot(q.T, temp_point) * temp_label) / beta
                c = (beta * b) / (np.linalg.norm(temp_point) ** 2)
                q = q.T + c * temp_point.T.dot(temp_label)

    return q


def plot_const_incr(data, clusters, epochs=10, beta=0.1):
    print('Spouští se upravená metoda konstantních přírůstků...')
    q = const_incr(clusters, epochs, beta)
    xmin, xmax = np.min(data[:, 0]), np.max(data[:, 0])
    ymin, ymax = np.min(data[:, 1]), np.max(data[:, 1])
    x = np.linspace(xmin, xmax, 2)
    y = []
    q = np.asarray(q)
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
    meshgrid_labels = classify(meshgrid_points, q)
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
