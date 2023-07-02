#!/usr/bin/env python
from caiman.base.movies import load, load_iter
from caiman.paths import caiman_datadir
import numpy as np


def test_load_iter():
    movies_directory = caiman_datadir()
    example_movie_path = os.path.join(movies_directory, 'example_movies', 'demoMovie.tif')

    subindices_list = [None, slice(100, None, 2)]

    for subindices in subindices_list:
        movie_iter = load_iter(example_movie_path, subindices=subindices)
        sum_movie_iter = calculate_movie_sum(movie_iter)
        
        expected_sum = load(example_movie_path, subindices=subindices).sum()
        assert np.isclose(sum_movie_iter, expected_sum, rtol=1e-6)


def calculate_movie_sum(movie_iter):
    movie_sum = 0
    try:
        while True:
            movie_sum += np.sum(next(movie_iter))
    except StopIteration:
        return movie_sum


if __name__ == '__main__':
    test_load_iter()