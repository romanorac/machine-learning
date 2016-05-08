# coding=utf8
import numpy as np


def rbf_distance(s1, s2, **kwargs):
    """
    calculate radial basis function

    :param s1: first element
    :param s2: second element
    :param kwargs: override sigma
    :return: distance between elements
    """

    def euclidean_dist(el1, el2, squared=True):
        """
        calculate euclidean distance
        :param el1: first element
        :param el2: second element
        :param squared: do not srqt distance
        :return:
        """
        dist = np.sum(np.square(el1 - el2))
        return dist if squared else np.sqrt(dist)

    sigma = kwargs.get("sigma", 1.0)
    gamma = 1 / float(2 * sigma ** 2)
    return np.exp(-gamma * euclidean_dist(s1, s2))


def levenshtein_distance(s1, s2, **kwargs):
    """
    calculate levenshtein distance

    one of properties of levenshtein distance is:
    - distance is at most the length of the longer string
    so we can normalize it with longer string.

    :param s1: first element
    :param s2: second element
    :param kwargs: normalize

    :return: distance between elements. if normalize is
    True, score is [0, 1], where 0 means equal string and
    1 means totally different.
    """
    normalize = kwargs.get("normalize", True)
    n1 = len(s1) + 1
    n2 = len(s2) + 1

    x = np.arange(n2)
    y = np.zeros((n2,), dtype=int)
    for i in np.arange(1, n1):
        c1 = s1[i - 1]
        y[0] = i
        for j in np.arange(1, n2):
            if c1 == s2[j - 1]:
                y[j] = x[j - 1]
            else:
                y[j] = min(x[j] + 1, y[j - 1] + 1, x[j - 1] + 1)
        x, y = y, x
    dist = x[n2 - 1]
    if normalize:
        dist /= float(max(len(s1), len(s2)))
    return dist


def distance_matrix(x, measure, **kwargs):
    """
    calculate distance matrix with given measure

    :param x: train samples
    :param measure: distance measure
    :param kwargs: optional arguments
    :return: distance matrix
    """
    n = len(x)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d[i][j] = d[j][i] = measure(x[i], x[j], **kwargs)
    return d
