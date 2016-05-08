from collections import Counter

import numpy as np


def info_gain(x, y, ft, separate_max):
    return info_gain_nominal(x, y, separate_max) if ft == "d" else info_gain_numeric(x, y)


def nominal_splits(x, y, x_vals, y_dist, separate_max):
    """
    Function uses heuristic to find best binary split of nominal values. Heuristic is described in (1) and it is
    originally defined for binary classes. We extend it to work with multiple classes by comparing label with least
    samples to others.

    x: numpy array - nominal feature
    y: numpy array - label
    x_vals: numpy array - unique nominal values of x
    y_dist: dictionary - distribution of labels

    Reference:
    (1) Classification and Regression Trees by Breiman, Friedman, Olshen, and Stone, pages 101- 102.
    """
    # select a label with least samples
    y_val = max(y_dist, key=y_dist.get) if separate_max else min(y_dist, key=y_dist.get)

    prior = y_dist[y_val] / float(len(y))  # prior distribution of selected label

    values, dist, splits = [], [], []
    for x_val in x_vals:  # for every unique nominal value
        dist.append(Counter(y[x == x_val]))  # distribution of labels at selected nominal value
        splits.append(x_val)
        suma = sum([prior * dist[-1][y_key] for y_key in y_dist.keys()])
        # estimate probability
        values.append(prior * dist[-1][y_val] / float(suma))
    indices = np.array(values).argsort()[::-1]

    # distributions and splits are sorted according to probabilities
    return np.array(dist)[indices], np.array(splits)[indices].tolist()


def h(values):
    """
    Function calculates entropy.

    values: list of integers
    """
    ent = np.true_divide(values, np.sum(values))
    return -np.sum(np.multiply(ent, np.log2(ent)))


def info_gain_nominal(x, y, separate_max):
    """
    Function calculates information gain for discrete features. If feature is continuous it is firstly discretized.

    x: numpy array - numerical or discrete feature
    y: numpy array - labels
    ft: string - feature type ("c" - continuous, "d" - discrete)
    split_fun: function - function for discretization of numerical features
    """
    x_vals = np.unique(x)  # unique values
    if len(x_vals) < 3:  # if there is just one unique value
        return None
    y_dist = Counter(y)  # label distribution
    h_y = h(y_dist.values())  # class entropy

    # calculate distributions and splits in accordance with feature type

    dist, splits = nominal_splits(x, y, x_vals, y_dist, separate_max)

    indices, repeat = (range(1, len(dist)), 1) if len(dist) < 50 else (range(1, len(dist), len(dist) / 10), 3)
    interval = len(dist) / 10

    max_ig, max_i, iteration = 0, 1, 0
    while iteration < repeat:
        for i in indices:
            dist0 = np.sum([el for el in dist[:i]])  # iter 0: take first distribution
            dist1 = np.sum([el for el in dist[i:]])  # iter 0: take the other distributions without first
            coef = np.true_divide([np.sum(dist0.values()), np.sum(dist1.values())], len(y))
            ig = h_y - np.dot(coef, [h(dist0.values()), h(dist1.values())])  # calculate information gain
            if ig > max_ig:
                max_ig, max_i = ig, i  # store index and value of maximal information gain
        iteration += 1
        if repeat > 1:
            interval = int(interval * 0.5)
            if max_i in indices and interval > 0:
                middle_index = indices.index(max_i)
            else:
                break
            min_index = middle_index if middle_index == 0 else middle_index - 1
            max_index = middle_index if middle_index == len(indices) - 1 else middle_index + 1
            indices = range(indices[min_index], indices[max_index], interval)

    # store splits of maximal information gain in accordance with feature type
    return float(max_ig), [splits[:max_i], splits[max_i:]]


def info_gain_numeric(x, y):
    x_unique = list(np.unique(x))
    if len(x_unique) == 1:
        return None
    indices = x.argsort()  # sort numeric attribute
    x, y = x[indices], y[indices]  # save sorted features with sorted labels

    right_dist = Counter(y)
    left_dist = Counter()

    diffs = np.nonzero(y[:-1] != y[1:])[0] + 1  # different neighbor classes have value True
    intervals = np.array((np.concatenate(([0], diffs[:-1])), diffs)).T

    if len(diffs) < 2:
        return None

    max_ig, max_i, max_j = 0, 0, 0
    prior_h = h(right_dist.values())  # calculate prior entropy

    for i, j in intervals:
        dist = Counter(y[i:j])
        left_dist += dist
        right_dist -= dist
        coef = np.true_divide((np.sum(left_dist.values()), np.sum(right_dist.values())), len(y))
        ig = prior_h - np.dot(coef, [h(left_dist.values()), h(right_dist.values())])
        if ig > max_ig:
            max_ig, max_i, max_j = ig, i, j

    if x[max_i] == x[max_j]:
        ind = x_unique.index(x[max_i])
        mean = np.float32(np.mean((x_unique[1 if ind == 0 else ind - 1], x_unique[ind])))
    else:
        mean = np.float32(np.mean((x[max_i], x[max_j])))

    return float(max_ig), [mean, mean]
