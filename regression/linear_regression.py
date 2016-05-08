"""
Linear regression

References
----------
http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex2/ex2.html
"""
import matplotlib.pyplot as plt
import numpy as np

from datasets import load


class LinearRegression:
    def __init__(self, samples, targets):
        self.samples = samples
        self.targets = targets
        self.thetas = []
        self.predictions = []

    def fit(self):
        a = np.dot(self.samples.transpose(), self.samples)  # A = samples * samples
        b = np.dot(self.samples.transpose(), self.targets)  # b = samples * target
        self.thetas = np.linalg.lstsq(a, b)[0]  # A^(-1) * b

    def predict(self, test_samples):
        self.predictions = np.dot(test_samples, self.thetas)  # y = kx + n


if __name__ == '__main__':
    samples, targets, _, _, _ = load.ex2()
    plt.scatter(samples[:, 1], targets)  # plot training data

    lin_reg = LinearRegression(samples, targets)
    lin_reg.fit()
    lin_reg.predict(samples)  # predict training data

    plt.plot(samples[:, 1], lin_reg.predictions)  # plot a line
    plt.show()
