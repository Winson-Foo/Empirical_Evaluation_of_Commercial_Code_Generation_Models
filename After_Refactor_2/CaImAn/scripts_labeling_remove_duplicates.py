#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Wed Jun 28 15:27:24 2017
@author: epnevmatikakis
"""

import os
import glob
from shutil import copyfile
import numpy as np
import caiman as cm
from caiman.base.rois import detect_duplicates, nf_merge_roi_zip, nf_read_roi_zip


# Constants
FOLDER_NAMES = [
    '/mnt/ceph/neuro/labeling/J115_2015-12-09_L01_ELS',
    '/mnt/ceph/neuro/labeling/J123_2015-11-20_L01_0',
    '/mnt/ceph/neuro/labeling/Jan-AMG_exp3_001',
    '/mnt/ceph/neuro/labeling/k37_20160109_AM_150um_65mW_zoom2p2_00001_1-16',
    '/mnt/ceph/neuro/labeling/k53_20160530',
    '/mnt/ceph/neuro/labeling/neurofinder.00.00',
    '/mnt/ceph/neuro/labeling/neurofinder.02.00',
    '/mnt/ceph/neuro/labeling/neurofinder.04.00',
    '/mnt/ceph/neuro/labeling/neurofinder.03.00.test',
    '/mnt/ceph/neuro/labeling/neurofinder.04.00.test'
]

PROJECTIONS_FOLDER = '/mnt/ceph/neuro/labeling'
REGIONS_FOLDER = 'regions'
PROJECTIONS_FILENAME = 'correlation_image.tif'
ACTIVE_REGIONS_FILENAME = '*active*regions.zip'


def get_img_shape(folder):
    return cm.load(os.path.join(folder, PROJECTIONS_FILENAME)).shape


def process_folder(folder):
    current_folder = os.path.join('/mnt/ceph/neuro/labeling', folder, REGIONS_FOLDER)
    img_shape = get_img_shape(os.path.join(PROJECTIONS_FOLDER, folder))
    filenames = glob.glob(os.path.join(current_folder, ACTIVE_REGIONS_FILENAME))
  
    for filename in filenames:
        ind_dup, ind_keep = detect_duplicates(filename, 0.25, FOV=img_shape)
        rois = nf_read_roi_zip(filename, img_shape)
        new_filename = filename[:-4] + '_nd.zip'
        print(filename)
        
        if not ind_dup:
            copyfile(filename, new_filename)
        else:
            nf_merge_roi_zip([filename], [ind_dup], filename[:-4] + '_copy')
            nf_merge_roi_zip([filename], [ind_keep], new_filename[:-4])
            print('FOUND!!')


def main():
    for folder in FOLDER_NAMES:
        process_folder(folder)


if __name__ == '__main__':
    main()