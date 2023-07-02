import numpy as np
import numpy.testing as npt
import os
from caiman.base.movies import load, load_iter
from caiman.paths import caiman_datadir


def calculate_sum_of_frames(movie_iter):
    S = 0
    while True:
        try:
            S += np.sum(next(movie_iter))
        except StopIteration:
            break
    return S


def test_load_iter():
    fname = os.path.join(caiman_datadir(), 'example_movies', 'demoMovie.tif')
    subindices_list = [None, slice(100, None, 2)]
    for subindices in subindices_list:
        movie_iter = load_iter(fname, subindices=subindices)
        S = calculate_sum_of_frames(movie_iter)
        expected_sum = load(fname, subindices=subindices).sum()
        npt.assert_allclose(S, expected_sum, rtol=1e-6)