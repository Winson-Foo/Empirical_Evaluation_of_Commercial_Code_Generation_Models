#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 10:09:09 2016
@author: agiovann
"""

from glob import glob
import scipy.stats as st
import calblitz as cb
import numpy as np

def save_image(m, name):
    img = cb.movie(m, fr=1)
    img.save(name)

def process_image(fl):
    m = cb.load(fl, fr=3)

    img = m.local_correlations(eight_neighbours=True)
    save_image(img, fl[:-4] + 'correlation_image.tif')

    m = np.array(m)

    img = st.skew(m, 0)
    save_image(img, fl[:-4] + 'skew.tif')

    img = st.kurtosis(m, 0)
    save_image(img, fl[:-4] + 'kurtosis.tif')

    img = np.std(m, 0)
    save_image(img, fl[:-4] + 'std.tif')

    img = np.median(m, 0)
    save_image(img, fl[:-4] + 'median.tif')

    img = np.max(m, 0)
    save_image(img, fl[:-4] + 'max.tif')


if __name__ == "__main__":
    for fl in glob('k36*compress_.tif'):
        print(fl)
        process_image(fl)

    process_image('All.tif')

    process_image('All_BL.tif')

    process_image('k31_20160104_MMA_150um_65mW_zoom2p2_00001_000_All.tif')

    process_image('k31_20160104_MMA_150um_65mW_zoom2p2_00001_000_All_BL.tif')