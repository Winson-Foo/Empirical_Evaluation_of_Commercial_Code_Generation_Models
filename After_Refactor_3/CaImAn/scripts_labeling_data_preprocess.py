#!/usr/bin/env python

import os
import scipy.stats as st
import calblitz as cb
from glob import glob
import numpy as np

def load_movie(folder_in, f_rate):
    fname_mov = os.path.join(os.path.split(folder_in)[0], os.path.split(folder_in)[-1] + 'MOV.hdf5')
    files = sorted(glob(os.path.join(os.path.split(folder_in)[0], 'images/*.tif')))
    m = cb.load_movie_chain(files, fr=f_rate)
    m.file_name = [os.path.basename(ttt) for ttt in m.file_name]
    m.save(fname_mov)
    del m
    return fname_mov

def create_images_for_labeling(f_name):
    cdir = os.path.dirname(f_name)

    m = cb.load(f_name)
    img = m.local_correlations(eight_neighbours=True)
    im = cb.movie(img, fr=1)
    im.save(os.path.join(cdir, 'correlation_image.tif'))

    img = np.std(m, 0)
    im = cb.movie(np.array(img), fr=1)
    im.save(os.path.join(cdir, 'std_projection.tif'))

    m1 = m.resize(1, 1, 1. / m.fr)
    img = np.median(m1, 0)
    im = cb.movie(np.array(img), fr=1)
    im.save(os.path.join(cdir, 'median_projection.tif'))

    m1 = m1 - img
    m1.save(os.path.join(cdir, 'MOV_BL.tif'))
    m1 = m1.bilateral_blur_2D()
    m1.save(os.path.join(cdir, 'MOV_BL_BIL.tif'))
    m = np.array(m1)

    img = np.max(m, 0)
    im = cb.movie(np.array(img), fr=1)
    im.save(os.path.join(cdir, 'max_projection.tif'))

    img = st.skew(m, 0)
    im = cb.movie(img, fr=1)
    im.save(os.path.join(cdir, 'skew_projection.tif'))
    del m
    del m1

if __name__ == "__main__":
    folders = [
        '/mnt/ceph/neuro/VariousLabeling/non_labelled_neurofinder/neurofinder.01.01.test/'
    ]
    f_rates = [7.5]

    for folder_in, f_rate in zip(folders, f_rates):
        f_name = load_movie(folder_in, f_rate)
        create_images_for_labeling(f_name)