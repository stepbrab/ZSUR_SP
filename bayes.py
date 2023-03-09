import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def get_params(clusters, data):
    # Spočítat průměr a rozptyl pro každou dimenzi pro každou třídu
    means = []
    variances = []
    for i in range(len(clusters)):
        for j in range(len(clusters[i])):
            means.append(np.mean(clusters[i][j], axis=0))
            variances.append(np.var(clusters[i][j], axis=0))
    return means, variances

def probability(x, mean, variance):
    # Vypočítat pravděpodobnost výskytu bodu v prostoru pro danou třídu
    exponent = -np.sum((x - mean)**2 / (2 * variance))
    return (1.0 / (np.sqrt(2 * np.pi * variance)) * np.exp(exponent))

def classify(x, means, variances):
    # Klasifikovat bod na základě nejvyšší pravděpodobnosti pro třídu
    probabilities = [probability(x, means[i], variances[i]) for i in range(len(means))]
    return np.argmax(probabilities)

