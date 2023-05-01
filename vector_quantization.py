import numpy as np
import matplotlib.pyplot as plt

#upravit strukturu, pochopit

def vq(data, num_iterations, clusters):
    # Inicializace kódové knihy jako náhodné podmnožiny trénovacích dat
    # codebook = data[np.random.choice(data.shape[0], codebook_size, replace=False)]
    codebook_size = len(clusters)
    codebook = np.zeros([3, 2])
    for i in range(codebook_size):
        codebook[i] = np.mean(clusters[i], axis=0)

    for i in range(num_iterations):
        # Nejbližší sousedé každého bodu v trénovacích datech od centroidu (codebook) asi nevims
        distances = np.linalg.norm(data[:, np.newaxis, :] - codebook, axis=2)
        closest_codes = np.argmin(distances, axis=1)

    return closest_codes, codebook


def vq_plot(data, num_iterations, clusters):
    codebook_size = len(clusters)

    codes, codebook = vq(data, 10, clusters)

    # Determination of range of space
    min_values = np.min(data, axis=0)
    max_values = np.max(data, axis=0)

    # Creation of meshgrid
    x_values, y_values = np.meshgrid(np.linspace(min_values[0], max_values[0], 100),
                                     np.linspace(min_values[1], max_values[1], 100))
    meshgrid = np.vstack([x_values.ravel(), y_values.ravel()]).T

    meshgrid_codes, _ = vq(meshgrid, 10, clusters)
    meshgrid_codes = meshgrid_codes.reshape(x_values.shape)


    plt.figure(figsize=(8, 8))
    plt.contourf(x_values, y_values, meshgrid_codes, alpha=0.2, levels=np.arange(codebook_size + 1) - 0.5,
                 cmap='jet')

    clusters = [data[codes == i] for i in range(codebook_size)]

    for i in range(codebook_size):
        plt.scatter(clusters[i][:, 0], clusters[i][:, 1])

    # Vykreslení kódové knihy jako červených křížků
    plt.scatter(codebook[:, 0], codebook[:, 1], marker='x', c='red')
    plt.show()
