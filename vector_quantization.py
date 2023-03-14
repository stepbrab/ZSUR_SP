import numpy as np
import matplotlib.pyplot as plt


def vq(data, codebook_size, num_iterations):
    # Inicializace kódové knihy jako náhodné podmnožiny trénovacích dat
    codebook = data[np.random.choice(data.shape[0], codebook_size, replace=False)]

    for i in range(num_iterations):
        # Nejbližší sousedé každého bodu v trénovacích datech
        distances = np.linalg.norm(data[:, np.newaxis, :] - codebook, axis=2)
        closest_codes = np.argmin(distances, axis=1)

        # Aktualizace kódové knihy jako průměr bodů přiřazených ke každému kódu
        for j in range(codebook_size):
            codebook[j] = np.mean(data[closest_codes == j], axis=0)

    return closest_codes, codebook