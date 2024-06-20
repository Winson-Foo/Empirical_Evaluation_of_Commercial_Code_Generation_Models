#!/usr/bin/env python

import numpy.testing as npt
import numpy as np
from caiman.source_extraction import cnmf


def make_G_matrix(g, T):
    """Create G matrix using given parameters.

    Args:
        g (np.ndarray): 1-D array of shape (N,) containing values for g.
        T (int): Total number of columns in the G matrix.

    Returns:
        np.matrix: G matrix of shape (N, T).

    """
    G = cnmf.temporal.make_G_matrix(T, g)
    return G.todense()


def test_make_G_matrix():
    """Test the make_G_matrix function."""
    g = np.array([1, 2, 3])
    T = 6
    G = make_G_matrix(g, T)

    # Expected G matrix
    true_G = np.matrix(
        [[1., 0., 0., 0., 0., 0.],
         [-1., 1., 0., 0., 0., 0.],
         [-2., -1., 1., 0., 0., 0.],
         [-3., -2., -1., 1., 0., 0.],
         [0., -3., -2., -1., 1., 0.],
         [0., 0., -3., -2., -1., 1.]])

    npt.assert_allclose(G, true_G)