import matplotlib.pyplot as plt
import numpy as np


def vq(data, num_iterations, clusters):
    codebook_size = len(clusters)
    codebook = np.zeros([3, 2])
    for i in range(codebook_size):
        codebook[i] = np.mean(clusters[i], axis=0)

    for i in range(num_iterations):
        distances = np.linalg.norm(data[:, np.newaxis, :] - codebook, axis=2)
        closest_codes = np.argmin(distances, axis=1)

    return closest_codes, codebook


def vq_plot(data, clusters):
    codebook_size = len(clusters)

    codes, codebook = vq(data, 10, clusters)

    min_values = np.min(data, axis=0)
    max_values = np.max(data, axis=0)

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
        plt.scatter(clusters[i][:, 0], clusters[i][:, 1], label=f"Shluk {i + 1}")

    plt.scatter(codebook[:, 0], codebook[:, 1], marker='x', c='k', label="Body kódové knihy")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title("Vektorová kvantizace")
    # plt.savefig("./pics/vq.eps", format='eps', dpi=300)
    plt.show()
