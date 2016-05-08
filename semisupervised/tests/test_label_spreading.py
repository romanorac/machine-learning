import unittest
from urlparse import urlparse

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
from sklearn.metrics.pairwise import rbf_kernel as sklearn_rbf_kernel
from sklearn.semi_supervised import LabelSpreading as LabelSpreadingSKLearn

from semisupervised.label_spreading import LabelSpreading as LabelSpreadingCustom
from semisupervised.measures import distance_matrix, rbf_distance, \
    levenshtein_distance


class ResultsComparison(unittest.TestCase):
    @staticmethod
    def test_levenshtein_distance():
        """
        compare the score of levenshtein distance
        """
        links = [
            "https://news.ycombinator.com/from?site=eff.org",
            "https://news.ycombinator.com/from?site=martinfowler.com",
            "https://news.ycombinator.com/news?p=2",
            "https://news.ycombinator.com/news?p=3"
        ]

        actual = distance_matrix(links, measure=levenshtein_distance,
                                 normalize=False)

        expected = np.array([[0, 14, 15, 15],
                             [14, 0, 24, 24],
                             [15, 24, 0, 1],
                             [15, 24, 1, 0]])
        assert_array_equal(actual, expected)

    @staticmethod
    def test_rbf_distance():
        """
        Compare radial basis function with scikit and our version
        """
        x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        expected = sklearn_rbf_kernel(x, gamma=0.5)

        actual = distance_matrix(x, measure=rbf_distance)
        n = len(actual)

        # scikits rbf returns 1 on diagonal as exp(0) = 1
        # so we change to 1 to assure equality
        actual[range(n), range(n)] = 1

        assert_almost_equal(actual, expected, decimal=20)

    @staticmethod
    def test_label_spreading_algorithms():
        """
        Compare scikit's algorithm and our algorithm
        """
        x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

        # scikit takes different input that our algorithm
        y_sklearn = np.array([1, 2, -1, -1])
        y_custom = np.array([[1, 0], [0, 1], [0, 0], [0, 0]])

        # scikit's algorithm
        alpha = 0.2
        max_iter = 30
        tol = 1e-3
        label_spreading = LabelSpreadingSKLearn(kernel="rbf",
                                                max_iter=max_iter,
                                                alpha=alpha, tol=tol)
        model = label_spreading.fit(x, y_sklearn)
        expected = model.predict(x)

        # our algorithm
        w = distance_matrix(x, measure=rbf_distance)
        ls = LabelSpreadingCustom(alpha=alpha, max_iter=max_iter, tol=tol)
        ls = ls.fit(w, y_custom)
        actual = ls.predict(y_custom)
        actual = np.array(actual) + 1  # add plus 1 to every prediction
        assert_array_equal(actual, expected)

    def test_pre_process_features(self):
        links = [
            "https://news.ycombinator.com/from?site=eff.org",
            "https://news.ycombinator.com/from?site=martinfowler.com",
            "https://news.ycombinator.com/news?p=2",
            "https://news.ycombinator.com/news?p=3"
        ]

        expected = ["/fromsite=eff.org",
                    "/fromsite=martinfowler.com",
                    "/newsp=2",
                    "/newsp=3"]

        actual = []
        for link in links:
            link_new = urlparse(link)
            actual.append(link_new.path + link_new.params + link_new.query)

        self.assertEqual(actual, expected)

    @staticmethod
    def test_levenshtein_normalize():
        links1 = [
            "https://news.ycombinator.com/from?site=eff.org",
            "https://news.ycombinator.com/from?site=martinfowler.com",
            "https://news.ycombinator.com/news?p=2",
            "https://news.ycombinator.com/news?p=3"
        ]

        links2 = ["/fromsite=eff.org",
                  "/fromsite=martinfowler.com",
                  "/newsp=2",
                  "/newsp=3"]

        actual_links1 = distance_matrix(links1, measure=levenshtein_distance,
                                        normalize=True)

        actual_links2 = distance_matrix(links2, measure=levenshtein_distance,
                                        normalize=True)

        expected_links1 = np.array([[0., 0.25454545, 0.32608696, 0.32608696],
                                    [0.25454545, 0., 0.43636364, 0.43636364],
                                    [0.32608696, 0.43636364, 0., 0.02702703],
                                    [0.32608696, 0.43636364, 0.02702703, 0.]])

        expected_links2 = np.array([[0., 0.53846154, 0.82352941, 0.82352941],
                                    [0.53846154, 0., 0.88461538, 0.88461538],
                                    [0.82352941, 0.88461538, 0., 0.125],
                                    [0.82352941, 0.88461538, 0.125, 0.]])

        assert_almost_equal(actual_links1, expected_links1, decimal=7)
        assert_almost_equal(actual_links2, expected_links2, decimal=7)
