import time

import numpy as np


# from main import dist_matrix

def dist_matrix(data):
    # Zkontrolovat
    # Výpočet matice vzdáleností
    dm = np.sqrt(np.sum((data[:, np.newaxis, :] - data[np.newaxis, :, :]) ** 2, axis=2))
    return dm


def div(data, index):
    st = time.time()

    firstIndex = index

    amount_of_classes = 1

    dm = dist_matrix(data)
    cutoff_dist = np.mean(dm)/3
    print(cutoff_dist)

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
        # print(values[i])
    for i in range(1, len(values) - 1):
        if values[i - 1] < values[i] > values[i + 1] and (values[i] - values[i - 1] - values[i + 1]) > cutoff_dist:
            amount_of_classes += 1

    return amount_of_classes
