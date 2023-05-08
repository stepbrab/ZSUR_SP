import numpy as np
import matplotlib.pyplot as plt
import numbers


def get_lines_constant_increment(data, beta):
    lines = _get_default_lines(data)
    # Zde pokračuje váš kód...


def get_lines_modified_constant_increment(data, beta):
    lines = _get_default_lines(data)
    # Zde pokračuje váš kód...


def generate_points(start, end, step):
    if isinstance(step, numbers.Real):
        data = [(i, j) for i in np.arange(start, end, step) for j in np.arange(start, end, step)]
    else:
        data = [(i, j) for i in range(start, end, step) for j in range(start, end, step)]
    return data


def merge_dicts(dict1, dict2):
    for k, v in dict2.items():
        if not v:
            continue
        else:
            dict1[k] = dict1[k] + v
    return dict1


def get_lines_ross(data):
    lines = _get_default_lines(data)
    # Zde pokračuje váš kód...


def _get_default_lines(data):
    lines = dict.fromkeys(data)
    for key in lines:
        lines[key] = list((1, 1, 1))
    return lines


def representatives(data, lines):
    vzory = dict.fromkeys(data)
    for key in data.keys():
        tmp = [1, 1, 1]
        x = data[key][0]
        no = 0
        for k in lines.keys():
            tmp[no] = 1 if x[1] > (lines[k][0] + x[0] * lines[k][1]) / -lines[k][2] else 0
            no += 1
        vzory[key] = tmp
    return vzory


def update_list_dict(d: dict, key, value):
    """Side effect function!"""
    if d[key] is None:
        d[key] = [value]
    else:
        d[key].append(value)


def classify_ross(lines, trypoints, repre):
    classified = dict.fromkeys(lines)
    for k in trypoints:
        stavajici = dict.fromkeys(lines)
        for j in lines.keys():
            if k[1] > (lines[j][0] + k[0] * lines[j][1]) / -lines[j][2]:
                stavajici[j] = 1
            else:
                stavajici[j] = 0
        for j in repre.keys():
            if repre[j] == list(stavajici.values()):
                update_list_dict(classified, j, k)
    return classified


def rosenblatt(data, space_size=(-20, 20), step=1):
    lines = get_lines_ross(data)
    # plot_lines(lines)  # needs some fixing, but algorithm is ok
    repre = representatives(data, lines)
    trypoints = generate_points(space_size[0], space_size[1], step)
    classified = classify_ross(lines, trypoints, repre)
    data = merge_dicts(data, classified)
    return data


def constant_increment(data, beta, space_size=(-20, 20), step=1):
    lines = get_lines_constant_increment(data, beta)
    repre = representatives(data, lines)
    trypoints = generate_points(space_size[0], space_size[1], step)
    classified = classify_ross(lines, trypoints, repre)
    data = merge_dicts(data, classified)
    return data


def modified_constant_increment(data, beta, space_size=(-20, 20), step=1):
    lines = get_lines_modified_constant_increment(data, beta)
    repre = representatives(data, lines)
    trypoints = generate_points(space_size[0], space_size[1], step)
    classified = classify_ross(lines, trypoints, repre)
    data = merge_dicts(data, classified)
    return data


def plot_lines(lines):
    t1 = np.arange(-20, 20, 1)
    for v in lines.values():
        plt.plot(t1, v[0] * t1 + v[1] * t1 + v[2])
    plt.ylim(-20, 20)
    plt.xlim(-20, 20)
    plt.show()
