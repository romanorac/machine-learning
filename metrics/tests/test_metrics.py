import unittest

import numpy.testing as npt

from metrics.gower_dissimilarity import gower_dissimilarity


class TestMetrics(unittest.TestCase):
    def test_gower_dissimilarity(self):
        """
        R CODE
        library(cluster)
        x1 <- c("brown", "blue", "red")
        x2 <- c("yellow","yellow","yellow")
        x3 <- c(1, 30, 20)
        x4 <- c(15, 12, 1)
        x <- data.frame(x1,x2,x3,x4)
        daisy(x, metric = "gower")
        """
        types = ["d", "d", "c", "c"]
        x1 = ["brown", "yellow", 1, 15]
        x2 = ["blue", "yellow", 30, 12]
        x3 = ["red", "yellow", 20, 1]
        ranges = [29, 14]

        sim_x1_x2 = gower_dissimilarity(x1, x2, types, ranges)
        sim_x1_x3 = gower_dissimilarity(x1, x3, types, ranges)
        sim_x2_x3 = gower_dissimilarity(x2, x3, types, ranges)

        npt.assert_almost_equal(sim_x1_x2, 0.5535714)
        npt.assert_almost_equal(sim_x1_x3, 0.6637931)
        npt.assert_almost_equal(sim_x2_x3, 0.5326355)
