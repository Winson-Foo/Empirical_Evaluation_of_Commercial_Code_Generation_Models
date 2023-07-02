#!/usr/bin/env python

import numpy.testing as npt
import numpy as np
from caiman.source_extraction import cnmf


def test_make_G_matrix():
    # Set up input variables
    coefficients = np.array([1, 2, 3])
    num_samples = 6

    # Calculate G matrix
    G = cnmf.temporal.make_G_matrix(num_samples, coefficients)
    G = G.todense()

    # Define the expected G matrix
    true_G = np.matrix(
        [[1., 0., 0., 0., 0., 0.],
         [-1., 1., 0., 0., 0., 0.],
         [-2., -1., 1., 0., 0., 0.],
         [-3., -2., -1., 1., 0., 0.],
         [0., -3., -2., -1., 1., 0.],
         [0., 0., -3., -2., -1., 1.]])

    # Compare calculated G matrix with expected G matrix
    npt.assert_allclose(G, true_G)