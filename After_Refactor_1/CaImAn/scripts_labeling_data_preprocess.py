#!/usr/bin/env python

import os
import numpy as np
import scipy.stats as st
import calblitz as cb
from glob import glob

def load_movie(file_path, frame_rate):
    files = sorted(glob(os.path.join(os.path.dirname(file_path), 'images/*.tif')))
    movie = cb.load_movie_chain(files, fr=frame_rate)
    movie.file_name = [os.path.basename(ttt) for ttt in movie.file_name]
    movie.save(file_path)
    del movie
    return file_path

def create_images_for_labeling(file_path):
    cdir = os.path.dirname(file_path)

    # Correlation image
    movie = cb.load(file_path)
    img = movie.local_correlations(eight_neighbours=True)
    im = cb.movie(img, fr=1)
    im.save(os.path.join(cdir, 'correlation_image.tif'))

    # Standard deviation image
    img = np.std(movie, 0)
    im = cb.movie(np.array(img), fr=1)
    im.save(os.path.join(cdir, 'std_projection.tif'))

    m1 = movie.resize(1, 1, 1. / movie.fr)

    # Median image
    img = np.median(m1, 0)
    im = cb.movie(np.array(img), fr=1)
    im.save(os.path.join(cdir, 'median_projection.tif'))

    # Save BL
    m1 = m1 - img
    m1.save(os.path.join(cdir, 'MOV_BL.tif'))
    m1 = m1.bilateral_blur_2D()
    m1.save(os.path.join(cdir, 'MOV_BL_BIL.tif'))
    m = np.array(m1)

    # Max image
    img = np.max(m, 0)
    im = cb.movie(np.array(img), fr=1)
    im.save(os.path.join(cdir, 'max_projection.tif'))

    # Skew image
    img = st.skew(m, 0)
    im = cb.movie(img, fr=1)
    im.save(os.path.join(cdir, 'skew_projection.tif'))
    del movie
    del m1

    return file_path

def process_files(params):
    file_paths = []
    for folder_in, f_rate in params:
        file_path = load_movie(os.path.join(os.path.dirname(folder_in), os.path.basename(folder_in) + 'MOV.hdf5'), f_rate)
        file_paths.append(file_path)
    
    results = list(map(create_images_for_labeling, file_paths))
    return results

if __name__ == "__main__":
    params = [
        ['/mnt/ceph/neuro/labeling/neurofinder.01.01/', 7.5],
    ]
    
    results = process_files(params)