#!/usr/bin/env python
import os
import numpy as np
import numpy.testing as npt
from caiman.base.movies import load, load_iter
from caiman.paths import caiman_datadir

def test_load_iter_correctness():
    """
    Test the correctness of load_iter() function from caiman.base.movies module.
    """
    # Arrange
    movie_path = os.path.join(caiman_datadir(), 'example_movies', 'demoMovie.tif')
    subindices_options = [None, slice(100, None, 2)]

    # Act and Assert
    for subindices in subindices_options:
        with load_iter(movie_path, subindices=subindices) as m:
            S = 0
            while True:
                try:
                    S += np.sum(next(m))
                except StopIteration:
                    break
        expected_sum = load(movie_path, subindices=subindices).sum()
        npt.assert_allclose(S, expected_sum, rtol=1e-6)