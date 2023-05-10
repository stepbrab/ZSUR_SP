import matplotlib.pyplot as plt
import numpy as np


def vq_train(clusters):
    print('Probíhá trénování vektorové kvantizace...')
    codebook_size = len(clusters)
    codebook = np.zeros([codebook_size, 2])
    for i in range(codebook_size):
        codebook[i] = np.mean(clusters[i], axis=0)
    return codebook


def vq_classify(data, codebook):
    print('Probíhá klasifikace pomocí vektorové kvantizace...')
    distances = np.linalg.norm(data[:, np.newaxis, :] - codebook, axis=2)
    codes = np.argmin(distances, axis=1)
    return codes


def vq_plot(data, clusters):
    codebook = vq_train(clusters)
    codebook_size = len(codebook)
    min_values = np.min(data, axis=0)
    max_values = np.max(data, axis=0)
    x_values, y_values = np.meshgrid(np.linspace(min_values[0], max_values[0], 100),
                                     np.linspace(min_values[1], max_values[1], 100))
    meshgrid = np.vstack([x_values.ravel(), y_values.ravel()]).T
    meshgrid_codes = vq_classify(meshgrid, codebook)
    meshgrid_codes = meshgrid_codes.reshape(x_values.shape)
    plt.figure(figsize=(8, 8))
    plt.contourf(x_values, y_values, meshgrid_codes, alpha=0.2, levels=np.arange(codebook_size + 1) - 0.5,
                 cmap='ocean')
    for i in range(codebook_size):
        plt.scatter(clusters[i][:, 0], clusters[i][:, 1], label=f"Shluk {i + 1}")
    plt.scatter(codebook[:, 0], codebook[:, 1], marker='x', c='k', label="Vektory kódové knihy")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title("Vektorová kvantizace")
    # plt.savefig("./pics/vq.eps", format='eps', dpi=300)
    plt.show()