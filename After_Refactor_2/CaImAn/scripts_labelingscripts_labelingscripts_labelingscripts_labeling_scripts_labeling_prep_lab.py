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

def process_image_file(file_path):
    m = cb.load(file_path, fr=3)

    img = m.local_correlations(eight_neighbours=True)
    im = cb.movie(img, fr=1)
    im.save(file_path[:-4] + 'correlation_image.tif')

    m = np.array(m)

    img = st.skew(m, 0)
    im = cb.movie(img, fr=1)
    im.save(file_path[:-4] + 'skew.tif')

    img = st.kurtosis(m, 0)
    im = cb.movie(img, fr=1)
    im.save(file_path[:-4] + 'kurtosis.tif')

    img = np.std(m, 0)
    im = cb.movie(img, fr=1)
    im.save(file_path[:-4] + 'std.tif')

    img = np.median(m, 0)
    im = cb.movie(img, fr=1)
    im.save(file_path[:-4] + 'median.tif')

    img = np.max(m, 0)
    im = cb.movie(img, fr=1)
    im.save(file_path[:-4] + 'max.tif')

# Process image files
for file_path in glob('k36*compress_.tif'):
    print(file_path)
    process_image_file(file_path)

# Process additional files
process_image_file('All.tif')
process_image_file('All_BL.tif')
process_image_file('k31_20160104_MMA_150um_65mW_zoom2p2_00001_000_All.tif')
process_image_file('k31_20160104_MMA_150um_65mW_zoom2p2_00001_000_All_BL.tif')