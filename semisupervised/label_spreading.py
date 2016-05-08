# coding=utf8
import numpy as np


class LabelSpreading:
    def __init__(self, alpha=0.2, max_iter=30, tol=1e-3):
        """
        :param alpha: clamping factor between (0,1)
        :param max_iter: maximum number of iterations
        :param tol: convergence tolerance
        """
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.dist = None

    def fit(self, w, y):
        """
        fit label spreading algorithm

        :param w: similarity matrix of n x n shape with n samples
        :param y: labels of n x c shape with c labels, where 1
         denotes label of x_i or 0 otherwise. Unlabeled samples
         have labels set to 0.
        """
        if type(w) != np.ndarray or type(y) != np.ndarray or len(w) != len(y):
            raise Exception("w and y should be numpy array with equal length")

        if 0 > self.alpha > 1 or self.max_iter < 0 or self.tol < 0:
            raise Exception("Parameters are set incorrectly")

        # construct the matrix S
        d = np.sum(w, axis=1)
        d[d == 0] = 1
        np.power(d, -1 / 2., d)
        d = np.diag(d)
        s = np.dot(np.dot(d, w), d)

        # Iterate F(t+1) until convergence
        cur_iter = 0
        err = self.tol
        f0 = y
        f1 = None
        while cur_iter < self.max_iter and err >= self.tol:
            f1 = self.alpha * np.dot(s, f0) + (1 - self.alpha) * y
            err = np.max(np.abs(f1 - f0))
            f0 = f1
            cur_iter += 1
        self.dist = f1  # set distributions
        return self

    def predict(self, y):
        """
        use model to create predictions

        :param y: labels of n x c shape with c labels, where 1
         denotes label of x_i or 0 otherwise. Unlabeled samples
         have labels set to 0.
        :return: list with predictions
        """
        if not np.any(y):
            raise Exception("Please fit model first")
        if type(y) != np.ndarray:
            raise Exception("y should be numpy array")
        predictions = []
        for i, labels in enumerate(y):
            index = np.where(labels == 1)[0]
            if len(index) == 1:
                # was labeled before
                predictions.append(index[0])
            else:
                # use label with highest score
                predictions.append(np.argmax(self.dist[i]))
        return predictions
