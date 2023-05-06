import time

import numpy as np
from matplotlib import pyplot as plt


# asi nejlepsi vec zatim

def dist_matrix(data):
    # Zkontrolovat
    # Výpočet matice vzdáleností
    dm = np.sqrt(np.sum((data[:, np.newaxis, :] - data[np.newaxis, :, :]) ** 2, axis=2))
    return dm


def div_plot_get_clusters(data, index, cutoff_dist):
    st = time.time()

    firstIndex = index

    indexes = []

    amount_of_classes = 1

    dm = dist_matrix(data)

    np.fill_diagonal(dm, np.inf)  # set diagonal to infinity

    values = np.zeros(len(data) - 1)

    for i in range(0, len(dm[0]) - 1):
        if i % 2 == 0:
            temp = dm[:][index]

        else:
            temp = dm[index][:]

        temp = temp[np.nonzero(temp)]

        values[i] = min(temp)

        if i % 2 == 0:
            indexDel = index
            index = np.where(dm[index][:] == values[i])[0][0]
            for x in range(0, len(dm[index])):
                dm[x][indexDel] = np.inf
                dm[indexDel][x] = np.inf


        else:
            indexDel = index
            index = np.where(dm[:][index] == values[i])[0][0]
            for x in range(0, len(dm[index])):
                dm[indexDel][x] = np.inf
                dm[x][indexDel] = np.inf

        indexes.append([index])

    plot_x_values = []
    plot_y_values = []
    cutoff_x_values = []
    cutoff_y_values = []

    plt.figure(figsize=(8, 8))
    # plt.title('Metoda řetězové mapy')
    for i in range(0, len(values)):
        if values[i] > cutoff_dist:
            amount_of_classes += 1
            cutoff_x_values.append([data[:][indexes[i - 1]][0][0], data[:][indexes[i]][0][0]])
            cutoff_y_values.append([data[:][indexes[i - 1]][0][1], data[:][indexes[i]][0][1]])

    for i in range(0, len(values)):
        plot_x_values.append(data[:][indexes[i]][0][0])
        plot_y_values.append(data[:][indexes[i]][0][1])

    plt.plot(plot_x_values, plot_y_values)
    plt.scatter(plot_x_values[0], plot_y_values[0], marker="x", color="k", label="Start bod")
    plt.scatter(plot_x_values[len(plot_x_values) - 1], plot_y_values[len(plot_y_values) - 1], marker="x", color="r",
                label="End bod")

    for i in range(0, len(cutoff_x_values)):
        plt.plot(cutoff_x_values[i], cutoff_y_values[i], label=f"Cutoff vzdálenost {i + 1}")
    plt.legend()
    plt.savefig("./pics/div.eps", format='eps', dpi=300)
    plt.show()

    return amount_of_classes
