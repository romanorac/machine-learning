import unittest
from operator import itemgetter

import numpy as np

from datasets import load
from ensemble import measures


class TestMeasures(unittest.TestCase):
    def test_margin1(self):
        y_dist = {"a": 0.6, "b": 0.3, "c": 0.1}
        y_test = ["a", "c"]

        probabilities = sorted(zip(y_dist.keys(), np.true_divide(y_dist.values(), np.sum(y_dist.values()))),
                               key=itemgetter(1),
                               reverse=True)
        prediction = max(y_dist, key=y_dist.get)

        expected_margins = [0.3, -0.5]
        for i, y in enumerate(y_test):
            if prediction == y:
                margin = probabilities[0][1] - probabilities[1][1] if len(probabilities) > 1 else 1
            else:
                margin = dict(probabilities).get(y, 0) - probabilities[0][1]

            self.assertEqual(expected_margins[i], margin)

    def test_margin2(self):
        y_dist = {"a": 1}
        y_test = ["a", "b"]

        probabilities = sorted(zip(y_dist.keys(), np.true_divide(y_dist.values(), np.sum(y_dist.values()))),
                               key=itemgetter(1),
                               reverse=True)
        prediction = max(y_dist, key=y_dist.get)

        expected_margins = [1, -1]
        for i, y in enumerate(y_test):
            if prediction == y:
                margin = probabilities[0][1] - probabilities[1][1] if len(probabilities) > 1 else 1
            else:
                margin = dict(probabilities).get(y, 0) - probabilities[0][1]

            self.assertEqual(expected_margins[i], margin)

    def test_nominal_info_gain(self):
        x, y, t, feature_names, name, dataset_type = load.breast_cancer()
        actual = np.array([measures.info_gain(x[:, i], y, t[i], True)[0] for i in range(len(x[0]))])

        expected = np.array(
            [0.363243, 0.588919, 0.569025, 0.375882, 0.494187, 0.520238, 0.490311, 0.457238, 0.199398])
        np.testing.assert_allclose(expected, actual, rtol=1e-05)
