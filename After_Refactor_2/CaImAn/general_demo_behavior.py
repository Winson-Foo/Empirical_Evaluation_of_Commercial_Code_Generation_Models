#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 11:43:39 2017
@author: agiovann
"""

import cv2
import logging
import numpy as np
import pylab as pl

import caiman as cm
from caiman.behavior import behavior
from caiman.utils.utils import download_demo

# Set up the logger; change this if you like.
logging.basicConfig(format="%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s] [%(process)d] %(message)s",
                    level=logging.WARNING)

def main():
    pl.ion()
    fname = ['demo_behavior.h5']
    
    if fname[0] in ['demo_behavior.h5']:
        fname = [download_demo(fname[0])]
    
    m = cm._load_behavior(fname[0])
    
    m = preprocess_behavior_movie(m)
    
    mask = select_roi_mask(m)
    
    n_components = 4
    resize_fact = 0.5
    num_std_mag_for_angle = 0.6
    only_magnitude = False
    method_factorization = 'dict_learn'
    max_iter_DL = -30
    
    spatial_filter_, time_trace_, of_or = cm.behavior.behavior.extract_motor_components_OF(m, n_components, mask=mask,
                                                                                            resize_fact=resize_fact, only_magnitude=only_magnitude,
                                                                                            verbose=True, method_factorization=method_factorization,
                                                                                            max_iter_DL=max_iter_DL)
    
    mags, dircts, dircts_thresh, spatial_masks_thrs = cm.behavior.behavior.extract_magnitude_and_angle_from_OF(
        spatial_filter_, time_trace_, of_or, num_std_mag_for_angle=num_std_mag_for_angle, sav_filter_size=3, only_magnitude=only_magnitude)
    
    plot_results(mags, dircts_thresh, spatial_filter_, m, mask)

def preprocess_behavior_movie(m):
    m = m.transpose([0, 2, 1])
    m = m[:, 150:, :]
    return m

def select_roi_mask(m):
    print("Please draw a polygon delimiting the ROI on the image that will be displayed after the image; press enter when done")
    mask = np.array(behavior.select_roi(np.median(m[::100], 0), 1)[0], np.float32)
    return mask

def plot_results(mags, dircts_thresh, spatial_filter_, m, mask):
    idd = 0
    axlin = pl.subplot(n_components, 2, 2)
    for mag, dirct, spatial_filter in zip(mags, dircts_thresh, spatial_filter_):
        pl.subplot(n_components, 2, 1 + idd * 2)
        min_x, min_y = np.min(np.where(mask), 1)

        spfl = spatial_filter
        spfl = cm.movie(spfl[None, :, :]).resize(
            1 / resize_fact, 1 / resize_fact, 1).squeeze()
        max_x, max_y = np.add((min_x, min_y), np.shape(spfl))

        mask[min_x:max_x, min_y:max_y] = spfl
        mask[mask < np.nanpercentile(spfl, 70)] = np.nan
        pl.imshow(m[0], cmap='gray')
        pl.imshow(mask, alpha=.5)
        pl.axis('off')

        axelin = pl.subplot(n_components, 2, 2 + idd * 2, sharex=axlin)
        pl.plot(mag / 10, 'k')
        dirct[mag < 0.5 * np.std(mag)] = np.nan
        pl.plot(dirct, 'r-', linewidth=2)

        idd += 1

if __name__ == "__main__":
    main()