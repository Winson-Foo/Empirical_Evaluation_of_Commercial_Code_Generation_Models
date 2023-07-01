#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 09:54:35 2017
"""

import os
import glob
import numpy as np
import pandas as pd
import pylab as pl
import cv2
import caiman as cm
from caiman.base.rois import nf_read_roi_zip
from caiman.source_extraction.cnmf import cnmf as cnmf

def load_cnmf_dataset(projection_img_correlation, folder_in, fl):
    """
    Load CNMF dataset

    Args:
        projection_img_correlation: Path to correlation image
        folder_in: Path to folder containing regions
        fl: Path to binary masks

    Returns:
        Cn: CNMF dataset
        roi_1: Loaded ROI from binary masks
        names_1: Loaded ROI names
    """
    Cn = cm.load(projection_img_correlation)
    shape = Cn.shape
    roi_1, names_1 = nf_read_roi_zip(fl, shape, return_names=True)
    return Cn, roi_1, names_1

def compare_labelers(folder_in, fl1, fl2, projection_img_correlation):
    """
    Compare labelers using binary masks

    Args:
        folder_in: Path to folder containing regions
        fl1: Path to first binary mask
        fl2: Path to second binary mask
        projection_img_correlation: Path to correlation image

    Returns:
        performance: Dictionary containing performance metrics
    """
    Cn, roi_1, roi_2 = load_cnmf_dataset(projection_img_correlation, folder_in, fl1, fl2)
    lab1, lab2 = fl1.split('/')[-1][:-4], fl2.split('/')[-1][:-4]
    tp_gt, tp_comp, fn_gt, fp_comp, performance = cm.base.rois.nf_match_neurons_in_binary_masks(roi_1, roi_2,
                                                                                                 thresh_cost=.7, min_dist=10,
                                                                                                 print_assignment=False,
                                                                                                 plot_results=False,
                                                                                                 Cn=Cn, labels=[lab1, lab2])
    performance['tp_gt'] = tp_gt
    performance['tp_comp'] = tp_comp
    performance['fn_gt'] = fn_gt
    performance['fp_comp'] = fp_comp
    return performance

def save_comparison_results(folder_in, performance_all):
    """
    Save comparison results to disk

    Args:
        folder_in: Path to folder containing regions
        performance_all: Dictionary containing all performance metrics
    """
    np.savez(folder_in + '/comparison_labelers_consensus.npz', performance_all=performance_all)

def process_folders(folders_out):
    """
    Process all folders

    Args:
        folders_out: List of folder paths

    Returns:
        df: DataFrame containing performance metrics
    """
    f1s = []
    names = []
    for folder_out in folders_out:
        projection_img_median = folder_out + '/projections/median_projection.tif'
        projection_img_correlation = folder_out + '/projections/correlation_image.tif'
        folder_in = folder_out + '/regions'
        print('********' + folder_out)
        with np.load(folder_in + '/comparison_labelers_consensus.npz', encoding='latin1') as ld:
            pf = ld['performance_all'][()]
            for key in pf:
                if pf[key]['f1_score'] <= 1:
                    print(str(pf[key]['f1_score'])[:5] + ':' +
                          str(key[-1]).split('/')[-1].split('_')[0])
                    f1s.append(pf[key]['f1_score'])
                    names.append(str(key[-1]).split('/')[-1].split('_')[0])
    df = pd.DataFrame({'names': names, 'f1s': f1s})
    g = df['f1s'].groupby(df['names'])
    return df

def plot_comparison_images(folders_out):
    """
    Plot comparison images for each folder

    Args:
        folders_out: List of folder paths
    """
    counter = 1
    for folder_out in folders_out:
        projection_img_median = folder_out + '/projections/median_projection.tif'
        projection_img_correlation = folder_out + '/projections/correlation_image.tif'
        folder_in = folder_out + '/regions'
        with np.load(folder_in + '/comparison_labelers.npz') as ld:
            pl.figure(figsize=(20, 10))
            counter += 1
            count = 0
            img = cm.load(projection_img_correlation)
            img[np.isnan(img)] = np.nanmean(img)
            for fl_m in [folder_in + '/*_matches.zip', folder_in + '/*1_mismatches.zip', folder_in + '/*0_mismatches.zip']:
                count += 1
                pl.subplot(1, 3, count)
                vmin, vmax = np.percentile(img, (5, 95))
                pl.imshow(img, vmin=vmin, vmax=vmax)
                rois_m = nf_read_roi_zip(glob.glob(fl_m)[0], img.shape)
                rois_m = rois_m.sum(0) * img.max()
                rois_m[rois_m == 0] = np.nan
                pl.imshow(rois_m, alpha=.5, cmap='hot', vmin=vmin, vmax=vmax)
                pl.axis('off')
                pl.xlabel(fl_m.split('/')[-1])

            pl.title(folder_out.split('/')[-1])
            pl.pause(1)

if __name__ == "__main__":
    # Set OpenCV to single thread
    try:
        cv2.setNumThreads(1)
    except Exception as e:
        print('OpenCV is naturally single threaded')

    # Check if running under iPython
    try:
        if __IPYTHON__:
            print((1))
            # this is used for debugging purposes only. allows to reload classes when changed
            get_ipython().magic('load_ext autoreload')
            get_ipython().magic('autoreload 2')
    except NameError:
        print('Not launched under iPython')

    # Set plot configuration
    pl.close('all')
    pl.rcParams['pdf.fonttype'] = 42
    font = {'family': 'Myriad Pro',
            'weight': 'regular',
            'size': 20}
    pl.rc('font', **font)

    folders_out = ['/mnt/ceph/neuro/labeling/J115_2015-12-09_L01_ELS',
                   '/mnt/ceph/neuro/labeling/J123_2015-11-20_L01_0',
                   '/mnt/ceph/neuro/labeling/Jan-AMG_exp3_001',
                   '/mnt/ceph/neuro/labeling/k37_20160109_AM_150um_65mW_zoom2p2_00001_1-16',
                   '/mnt/ceph/neuro/labeling/k53_20160530',
                   '/mnt/ceph/neuro/labeling/neurofinder.00.00',
                   '/mnt/ceph/neuro/labeling/neurofinder.02.00',
                   '/mnt/ceph/neuro/labeling/neurofinder.04.00',
                   '/mnt/ceph/neuro/labeling/neurofinder.03.00.test',
                   '/mnt/ceph/neuro/labeling/neurofinder.04.00.test',
                   '/mnt/ceph/neuro/labeling/neurofinder.01.01',
                   '/mnt/ceph/neuro/labeling/packer.001',
                   '/mnt/ceph/neuro/labeling/Yi.data.001',
                   '/mnt/ceph/neuro/labeling/yuste.Single_150u',
                   '/mnt/ceph/neuro/labeling/Jan-AMG1_exp2_new_001']

    for folder_out in folders_out:
        projection_img_median = folder_out + '/projections/median_projection.tif'
        projection_img_correlation = folder_out + '/projections/correlation_image.tif'
        folder_in = folder_out + '/regions'
        performance_all = dict()
        fls = list(glob.glob(folder_in + '/*_nd.zip'))
        consensus_counter = dict()
        fl1 = os.path.join(folder_in, 'joined_consensus_active_regions.zip')
        for fl2 in fls:
            print([fl1, fl2])
            performance = compare_labelers(folder_in, fl1, fl2, projection_img_correlation)
            performance_all[fl1, fl2] = performance

        save_comparison_results(folder_in, performance_all)

    df = process_folders(folders_out)
    plot_comparison_images(folders_out)