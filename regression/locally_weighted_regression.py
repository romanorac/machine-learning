"""
Locally Weighted Regression

References
----------
CS229 Lecture notes1, Chapter 3 Locally weighted linear regression, Prof. Andrew Ng
http://cs229.stanford.edu/notes/cs229-notes1.pdf

weighted least squares and locally weighted linear regression
http://www.dsplog.com/2012/02/05/weighted-least-squares-and-locally-weighted-linear-regression/
"""

import matplotlib.pyplot as plt
import numpy as np

from datasets import load


class LocallyWeightedRegression:
    def __init__(self, samples, targets):
        """
        :param samples: {array-like, sparse matrix}, shape = [n_samples, n_features]
                        Training vectors, where n_samples is the number of samples and
                        n_features is the number of features.
        :param targets: array-like, shape = [n_samples]
                        Target values.
        """
        self.samples = samples
        self.targets = targets
        self.thetas = []
        self.estimation = []

    def fit_predict(self, estimation_samples, tau=1):
        """
        fit estimation_samples with locally weighted regression according to samples and target

        :param estimation_samples : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Estimation vectors, where n_samples is the number of samples and
            n_features is the number of features.

        :param tau : float, tau >= 0
            The the bandwidth parameter tau controls how quickly the weight of a training example falls off with
             distance of its x(i) from the query point x.
        """
        if tau < 0:
            print "tau should be greater than 0"
            return [], []

        self.thetas = []
        self.estimation = []

        for x in estimation_samples:
            # calculate weights that depend on the particular vector x
            weights = np.exp((-(self.samples - x) * (self.samples - x)).sum(axis=1) / (2 * tau ** 2))
            diagonal_weights = np.diag(weights)  # diagonal matrix with weights
            samples_times_weights = np.dot(self.samples.T, diagonal_weights)
            a = np.dot(samples_times_weights, self.samples)
            b = np.dot(samples_times_weights, self.targets)

            self.thetas.append(np.linalg.lstsq(a, b)[0])  # calculate thetas for given x with: A^-1 * b
            self.estimation.append(np.dot(x, self.thetas[-1]))  # calculate estimation for given x and thetas


if __name__ == '__main__':
    samples, targets, _, _, _ = load.lwlr()
    plt.scatter(samples[:, 1], targets)  # plot train data

    lwr = LocallyWeightedRegression(samples, targets)

    taus = [1, 10, 25]
    color = ["r", "g", "b"]
    for i, tau in enumerate(taus):
        lwr.fit_predict(samples, tau=tau)
        plt.plot(samples[:, 1], lwr.estimation, c=color[i])  # plot estimations
    plt.show()
