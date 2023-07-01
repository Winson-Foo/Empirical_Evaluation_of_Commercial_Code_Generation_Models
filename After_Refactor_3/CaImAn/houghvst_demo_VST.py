import timeit
import numpy as np
import os
import caiman.external.houghvst.estimation as est
from caiman.external.houghvst.gat import compute_gat, compute_inverse_gat
import caiman as cm
from caiman.paths import caiman_datadir


def load_movie():
    fnames = [os.path.join(caiman_datadir(), 'example_movies', 'demoMovie.tif')]
    movie = cm.load(fnames)
    movie = movie.astype(float)
    # makes estimation numerically better:
    movie -= movie.mean()
    return movie


def estimate_vst_movie(movie_train, stride):
    t = timeit.default_timer()
    estimation_res = est.estimate_vst_movie(movie_train, stride=stride)
    print('\tTime', timeit.default_timer() - t)
    alpha = estimation_res.alpha
    sigma_sq = estimation_res.sigma_sq
    return alpha, sigma_sq


def compute_gat_movie(movie, sigma_sq, alpha):
    movie_gat = compute_gat(movie, sigma_sq, alpha=alpha)
    # save movie_gat here
    return movie_gat


def compute_inverse_gat_movie(movie_gat, sigma_sq, alpha, method='asym'):
    movie_gat_inv = compute_inverse_gat(movie_gat, sigma_sq, alpha=alpha, method=method)
    # save movie_gat_inv here
    return movie_gat_inv


def main():
    movie = load_movie()
    
    # use one every 200 frames
    temporal_stride = 200
    # use one every 8 patches (patches are 8x8 by default)
    spatial_stride = 8

    movie_train = movie[::temporal_stride]

    alpha, sigma_sq = estimate_vst_movie(movie_train, spatial_stride)

    movie_gat = compute_gat_movie(movie, sigma_sq, alpha)
    movie_gat_inv = compute_inverse_gat_movie(movie_gat, sigma_sq, alpha, method='asym')
    
    return movie, movie_gat_inv


movie, movie_gat_inv = main()

cm.concatenate([movie, movie_gat_inv], axis=1).play(gain=10, magnification=4)