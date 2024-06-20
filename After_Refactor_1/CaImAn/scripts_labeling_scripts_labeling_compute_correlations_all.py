#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Fri Jun  2 16:27:17 2017

@brief: This script performs motion correction on a set of image files.

@details:
- Imports OpenCV and other necessary libraries.
- Sets up environment related configurations.
- Performs motion correction on each image file.
- Saves the motion corrected results.

"""

import cv2
import os
import glob
import numpy as np
import pylab as pl
import caiman as cm

def main():
    """
    Main method to execute the motion correction analysis.
    """
    configure_opencv()
    configure_ipython()

    image_files = get_image_files()

    # Run analysis for each file
    for image_file in image_files:
        motion_correct_img(image_file)

def configure_opencv():
    """
    Configure OpenCV settings.
    """
    try:
        cv2.setNumThreads(1)
    except:
        print('OpenCV is naturally single threaded')

def configure_ipython():
    """
    Configure IPython settings.
    """
    try:
        if __IPYTHON__:
            print((1))
            # this is used for debugging purposes only. allows to reload classes
            # when changed
            get_ipython().magic('load_ext autoreload')
            get_ipython().magic('autoreload 2')
    except NameError:
        print('Not launched under iPython')

def get_image_files():
    """
    Get the list of image files to process.
    """
    base = '/mnt/ceph/neuro/labeling/'

    image_files = [
      'neurofinder.00.00',
      'neurofinder.02.00',
      'neurofinder.03.00.test',
      'neurofinder.04.00',
      'neurofinder.04.00.test',
      'neurofinder.01.01',
      'k37_20160109_AM_150um_65mW_zoom2p2_00001_1-16',
      'yuste.Single_150u',
      'packer.001',
      'J123_2015-11-20_L01_0',
      'Yi.data.001',
      'FN.151102_001',
      'Jan-AMG1_exp2_new_001',
      'Jan-AMG_exp3_001',
      'k31_20160107_MMP_150um_65mW_zoom2p2_00001_1-15',
      'k31_20160106_MMA_400um_118mW_zoom2p2_00001_1-19',
      'k36_20151229_MMA_200um_65mW_zoom2p2_00001_1-17'
    ]

    image_files = [os.path.join(base, image_file) for image_file in image_files]

    return image_files

def motion_correct_img(image_file):
    """
    Perform motion correction on an image file.
    """
    nms = glob.glob(os.path.join(image_file, 'images/*.tif'))
    nms.sort()

    # Remove files with '_BL' in the name
    nms = [nm for nm in nms if '_BL' not in nm]

    if len(nms) > 0:
        pl.subplot(5, 4, count)
        count += 1

        templ = cm.load(os.path.join(image_file, 'projections/median_projection.tif'))
        mov_tmp = cm.load(nms[0], subindices=range(400))

        if mov_tmp.shape[1:] != templ.shape:
            diffx, diffy = np.subtract(mov_tmp.shape[1:], templ.shape) // 2 + 1

        vmin, vmax = np.percentile(templ, 5), np.percentile(templ, 95)
        pl.imshow(templ, vmin=vmin, vmax=vmax)

        min_mov = np.nanmin(mov_tmp)
        mc_list = []
        mc_templs_part = []
        mc_templs = []
        mc_fnames = []

        for each_file in nms:
            # Perform motion correction using the MotionCorrect class
            mc = MotionCorrect(each_file, min_mov,
                               dview=dview, max_shifts=max_shifts, niter_rig=niter_rig, splits_rig=splits_rig,
                               num_splits_to_process_rig=num_splits_to_process_rig,
                               shifts_opencv=True, nonneg_movie=True)
            mc.motion_correct_rigid(template=templ, save_movie=True)
            new_templ = mc.total_template_rig

            pl.imshow(new_templ, cmap='gray')
            pl.pause(.1)

            mc_list += mc.shifts_rig
            mc_templs_part += mc.templates_rig
            mc_templs += [mc.total_template_rig]
            mc_fnames += mc.fname_tot_rig

        np.savez(os.path.join(image_file, 'images/mot_corr_res.npz'), mc_list=mc_list,
                 mc_templs_part=mc_templs_part, mc_fnames=mc_fnames, mc_templs=mc_templs)

    print([os.path.split(nm)[-1] for nm in nms])
    print([int(os.path.getsize(nm) / 1e+9 * 100) / 100. for nm in nms])

if __name__ == '__main__':
    main()