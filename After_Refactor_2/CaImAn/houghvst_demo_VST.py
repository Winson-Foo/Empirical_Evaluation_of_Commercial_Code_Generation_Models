import timeit
import os
import caiman as cm
from caiman.external.houghvst.estimation import estimate_vst_movie
from caiman.external.houghvst.gat import compute_gat, compute_inverse_gat
from caiman.paths import caiman_datadir

def load_movie():
    fnames = [os.path.join(caiman_datadir(), 'example_movies', 'demoMovie.tif')]
    movie = cm.load(fnames)
    movie = movie.astype(float)
    return movie

def preprocess_movie(movie):
    movie -= movie.mean()
    return movie

def estimate_vst(movie, temporal_stride, spatial_stride):
    movie_train = movie[::temporal_stride]
    t = timeit.default_timer()
    estimation_res = estimate_vst_movie(movie_train, stride=spatial_stride)
    print('\tTime', timeit.default_timer() - t)
    alpha = estimation_res.alpha
    sigma_sq = estimation_res.sigma_sq
    return alpha, sigma_sq, estimation_res

def compute_gat_movie(movie, sigma_sq, alpha):
    movie_gat = compute_gat(movie, sigma_sq, alpha=alpha)
    return movie_gat

def compute_inverse_gat_movie(movie_gat, sigma_sq, alpha):
    movie_gat_inv = compute_inverse_gat(movie_gat, sigma_sq, alpha=alpha, method='asym')
    return movie_gat_inv

def main():
    movie = load_movie()
    movie = preprocess_movie(movie)
    temporal_stride = 200
    spatial_stride = 8
    alpha, sigma_sq, estimation_res = estimate_vst(movie, temporal_stride, spatial_stride)
    movie_gat = compute_gat_movie(movie, sigma_sq, alpha)
    movie_gat_inv = compute_inverse_gat_movie(movie_gat, sigma_sq, alpha)
    return movie, movie_gat_inv

movie, movie_gat_inv = main()