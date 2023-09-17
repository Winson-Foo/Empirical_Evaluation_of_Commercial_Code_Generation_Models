import timeit
import numpy as np
import os
from caiman import load, concatenate
from caiman.paths import caiman_datadir
from caiman.external.houghvst.estimation import estimate_vst_movie
from caiman.external.houghvst.gat import compute_gat, compute_inverse_gat


def main():
    # Load the movie
    movie_path = os.path.join(caiman_datadir(), 'example_movies', 'demoMovie.tif')
    movie = load(movie_path).astype(float)

    # Subtract the mean to improve estimation
    movie -= movie.mean()

    # Set stride values
    temporal_stride = 200
    spatial_stride = 8

    # Create a training movie
    movie_train = movie[::temporal_stride]

    # Estimate parameters
    t = timeit.default_timer()
    estimation_res = estimate_vst_movie(movie_train, stride=spatial_stride)
    print('\tTime', timeit.default_timer() - t)

    # Extract alpha and sigma_sq parameters
    alpha = estimation_res.alpha
    sigma_sq = estimation_res.sigma_sq

    # Compute gaussian adaptive thresholding (GAT)
    movie_gat = compute_gat(movie, sigma_sq, alpha=alpha)
    save_movie_gat(movie_gat)

    # Compute inverse GAT
    movie_gat_inv = compute_inverse_gat(movie_gat, sigma_sq, alpha=alpha, method='asym')
    save_movie_gat_inv(movie_gat_inv)

    return movie, movie_gat_inv


def save_movie_gat(movie_gat):
    # Save movie_gat implementation here
    pass


def save_movie_gat_inv(movie_gat_inv):
    # Save movie_gat_inv implementation here
    pass


if __name__ == '__main__':
    movie, movie_gat_inv = main()
    concatenate([movie, movie_gat_inv], axis=1).play(gain=10, magnification=4)