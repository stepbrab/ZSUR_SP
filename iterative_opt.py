import numpy as np
from matplotlib import pyplot as plt


# to jsem psal ja, tady nejde o kopirovani ale spis to moc nefunguje lmao

def get_J(clusters):
    J = 0
    for i in range(len(clusters)):
        dists = np.sqrt(np.sum((clusters[i] - np.mean(clusters[i], axis=0)) ** 2))
        J += np.sum(dists ** 2)
    return J


def it_opt(clusters):  # presouva to i kdyz to J neni lepsi... nijak se nemeni grafy nevim
    J = get_J(clusters)
    s = [len(clusters[0]), len(clusters[1]), len(clusters[2])]
    s1 = s[0]
    s2 = s[1]
    s3 = s[2]
    print("Původní: ", J, ";; s hodnoty: ", s, np.sum(s))
    diff = 0
    for i in range(len(clusters)):
        for j in range(len(clusters)):
            if i != j:
                k = 0
                while k < len(clusters[i]):
                    # A1 = s[i] / (s[i]-1) * np.sqrt(
                    #     np.sum((clusters[i][k] - np.mean(clusters[i], axis=0)) ** 2))
                    # A2 = s[j] / (s[j]+1) * np.sqrt(
                    #     np.sum((clusters[i][k] - np.mean(clusters[j], axis=0)) ** 2))
                    # A1 = s[i] / (s[i]-1) * math.dist(clusters[i][k], np.mean(clusters[i], axis=0))
                    # A2 = s[j] / (s[j] + 1) * math.dist(clusters[i][k], np.mean(clusters[j], axis=0))
                    A1 = s[i] / (s[i] - 1) * np.linalg.norm(clusters[i][k] - np.mean(clusters[i], axis=0))
                    A2 = s[j] / (s[j] + 1) * np.linalg.norm(clusters[i][k] - np.mean(clusters[j], axis=0))
                    if A1 > A2:
                        # clusters[j].append(clusters[i][k])
                        clusters[j] = np.append(clusters[j], [clusters[i][k]], axis=0)
                        clusters[i] = np.delete(clusters[i], k, axis=0)
                        s[i] = len(clusters[i])
                        s[j] = len(clusters[j])
                        k -= 1
                        improved_J = get_J(clusters)
                        print("Nový: ", improved_J, ";; s hodnoty: ", s, np.sum(s))
                        diff += 1
                    k += 1
    plt.figure(figsize=(8, 8))
    i = 0
    for cluster in clusters:
        plt.scatter(cluster[:, 0], cluster[:, 1], label=f"Shluk {i + 1}")
        i += 1
    # plt.title("Iter_opt")
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig("./pics/iter_opt.eps", format='eps', dpi=300)
    plt.show()
    return clusters, diff
