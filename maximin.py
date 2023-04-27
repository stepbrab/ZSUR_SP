import numpy as np


def dist_matrix(data):
    # Zkontrolovat
    # Výpočet matice vzdáleností
    dm = np.sqrt(np.sum((data[:, np.newaxis, :] - data[np.newaxis, :, :]) ** 2, axis=2))
    return dm


# def mm(data, eps):
#     dm = dist_matrix(data)
#     n = dm.shape[0]
#     centers = [0]
#     while len(centers) < n:
#         print(len(centers))
#         d = np.min(dm[centers], axis=0)
#         i = np.argmax(d)
#         if d[i] > eps:
#             centers.append(i)
#         else:
#             break
#     return len(centers), centers

def mm(data, cutoff_dist):
    dm = dist_matrix(data)
    cutoff_dist = np.mean(dm) #pro data:10 funguje, data:5 nefunguje
    n = dm.shape[0]
    centers = [0]
    dist_to_centers = dm[0]
    while len(centers) < n:
        # najdeme nejvzdalenejsi bod od stredů shluků
        i = np.argmax(dist_to_centers)
        dist = dist_to_centers[i]
        if dist > cutoff_dist:
            centers.append(i)
            # aktualizujeme vzdálenosti pro nový střed
            new_d = dm[i]
            dist_to_centers = np.minimum(dist_to_centers, new_d)
        else:
            break
    return len(centers)
