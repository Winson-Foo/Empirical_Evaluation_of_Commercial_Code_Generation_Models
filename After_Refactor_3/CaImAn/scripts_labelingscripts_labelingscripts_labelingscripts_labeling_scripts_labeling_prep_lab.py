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

def process_file(file_path, prefix):
    m = cb.load(file_path, fr=3)

    img = m.local_correlations(eight_neighbours=True)
    im = cb.movie(img, fr=1)
    im.save(f"{prefix}_correlation_image.tif")

    m = np.array(m)

    img = st.skew(m, 0)
    im = cb.movie(img, fr=1)
    im.save(f"{prefix}_skew.tif")

    img = st.kurtosis(m, 0)
    im = cb.movie(img, fr=1)
    im.save(f"{prefix}_kurtosis.tif")

    img = np.std(m, 0)
    im = cb.movie(img, fr=1)
    im.save(f"{prefix}_std.tif")

    img = np.median(m, 0)
    im = cb.movie(img, fr=1)
    im.save(f"{prefix}_median.tif")

    img = np.max(m, 0)
    im = cb.movie(img, fr=1)
    im.save(f"{prefix}_max.tif")

file_paths = glob('k36*compress_.tif')
for file_path in file_paths:
    print(file_path)
    prefix = file_path[:-4]
    process_file(file_path, prefix)

process_file('All.tif', 'All')
process_file('All_BL.tif', 'All_BL')
process_file('k31_20160104_MMA_150um_65mW_zoom2p2_00001_000_All.tif', 'k31_20160104_MMA_150um_65mW_zoom2p2_00001_000')
process_file('k31_20160104_MMA_150um_65mW_zoom2p2_00001_000_All_BL.tif', 'k31_20160104_MMA_150um_65mW_zoom2p2_00001_000_BL')