#!/usr/bin/env python

import numpy.testing as npt
import numpy as np
from caiman.source_extraction import cnmf


def generate_true_G(g, T):
    """
    Generates the true G matrix for testing make_G_matrix function.
    Args:
        g: array-like, Temporal filter coefficients.
        T: int, Total number of time points.
    Returns:
        true_G: numpy matrix, The true G matrix.
    """
    true_G = np.zeros((T, T))
    for i in range(T):
        for j in range(i + 1):
            true_G[i, j] = g[i - j]
    return true_G


def test_make_G_matrix():
    """
    Tests the make_G_matrix function.
    """
    g = [1, 2, 3]
    T = 6
    G = cnmf.temporal.make_G_matrix(T, g)
    G = G.todense()

    true_G = generate_true_G(g, T)

    npt.assert_allclose(G, true_G)


if __name__ == "__main__":
    test_make_G_matrix()