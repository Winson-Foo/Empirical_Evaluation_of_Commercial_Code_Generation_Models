#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from PIL import Image
from ca_source_extraction import utilities as cse_utils


def extract_sue_ann_info(fname, base_folder):
    matvar = loadmat(fname)
    idx = -1
    for counter, nm in enumerate(matvar['roiFile'][0]):
        if nm[0][0] in os.path.split(os.path.dirname(base_folder))[-1]:
            idx = counter
            A = matvar['A_s'][0][idx]
            A_init = matvar['A_ins'][0][idx]
            C = matvar['C_s'][0][idx]
            template = matvar['templates'][0][idx]
            idx_shapes = matvar['init'][0][idx]
            idx_global = matvar['globalId'][0][idx]

    if idx < 0:
        raise Exception('Matching name not found!')

    return A, C, template, idx_shapes, A_init


def load_image(fname):
    return np.array(Image.open(fname))


def plot_images(img_corr, img_median, rois_1, rois_2, vmax_corr_perc, vmin_corr_perc, vmax_median_perc, vmin_median_perc):
    plt.figure(facecolor="white")

    # Correlation image subplot
    plt.subplot(2, 4, 1)
    plt.imshow(img_corr, cmap='gray', vmax=np.percentile(img_corr, vmax_corr_perc), vmin=np.percentile(img_corr, vmin_corr_perc))
    plt.imshow(np.nanmax(rois_1, -1), cmap='ocean', vmax=2, alpha=.2, vmin=0)
    plt.ylabel('CORR IMAGE')
    plt.title('PRINCETON')
    plt.axis('off')

    plt.subplot(2, 4, 2)
    plt.imshow(img_corr, cmap='gray', vmax=np.percentile(img_corr, vmax_corr_perc), vmin=np.percentile(img_corr, vmin_corr_perc))
    plt.imshow(np.nanmax(rois_2, -1), cmap='hot', alpha=.2, vmin=0, vmax=3)
    plt.title('BEN')
    plt.axis('off')

    plt.subplot(2, 4, 3)
    plt.imshow(img_corr, cmap='gray', vmax=np.percentile(img_corr, vmax_corr_perc), vmin=np.percentile(img_corr, vmin_corr_perc))
    plt.axis('off')

    # Median image subplot
    plt.subplot(2, 4, 5)
    plt.imshow(img_median, cmap='gray', vmax=np.percentile(img_median, vmax_median_perc), vmin=np.percentile(img_median, vmin_median_perc))
    plt.imshow(np.nanmax(rois_1, -1), cmap='ocean', vmax=2, alpha=.2, vmin=0)
    plt.ylabel('MEDIAN')
    plt.axis('off')

    plt.subplot(2, 4, 6)
    plt.imshow(img_median, cmap='gray', vmax=np.percentile(img_median, vmax_median_perc), vmin=np.percentile(img_median, vmin_median_perc))
    plt.imshow(np.nanmax(rois_2, -1), cmap='hot', alpha=.2, vmin=0, vmax=3)
    plt.axis('off')

    plt.subplot(2, 4, 7)
    plt.imshow(img_median, cmap='gray', vmax=np.percentile(img_median, vmax_median_perc), vmin=np.percentile(img_median, vmin_median_perc))
    plt.axis('off')

    # Combined subplot
    plt.subplot(2, 4, 4)
    plt.imshow(np.nanmax(rois_1, -1), cmap='ocean', vmax=2, alpha=.5, vmin=0)
    plt.imshow(np.nanmax(rois_2, -1), cmap='hot', alpha=.5, vmin=0, vmax=3)
    plt.axis('off')

    plt.subplot(2, 4, 8)
    plt.axis('off')
    plt.imshow(img_corr, cmap='gray', vmax=np.percentile(img_corr, vmax_corr_perc), vmin=np.percentile(img_corr, vmin_corr_perc))
    plt.imshow(np.nanmax(rois_1, -1), cmap='ocean', vmax=2, alpha=.3, vmin=0)
    plt.imshow(np.nanmax(rois_2, -1), cmap='hot', alpha=.3, vmin=0, vmax=3)
    plt.axis('off')

    font = {'family': 'Myriad Pro', 'weight': 'regular', 'size': 30}
    plt.rc('font', **font)

    plt.rcParams['pdf.fonttype'] = 42
    plt.savefig(base_folder + 'comparison.pdf')


def process_codebase(base_folder):
    img_corr = load_image(base_folder + 'projections/correlation_image.tif')
    img_median = load_image(base_folder + 'projections/median_projection.tif')
    shape = np.shape(img_median)

    if os.path.exists(base_folder + 'regions/princeton_regions.mat'):
        a = loadmat(base_folder + 'regions/princeton_regions.mat')
        try:
            rois_1 = a['allROIs']
        except:
            rois_1 = a['M']
    elif os.path.exists(base_folder + 'regions/sue_ann_regions.mat'):
        A, C, template, idx_shapes, A_in = extract_sue_ann_info(base_folder + 'regions/sue_ann_regions.mat', base_folder)
        rois_1 = np.reshape(A.todense(), (shape[0], shape[1], -1), order='F')
    else:
        rois_1 = np.transpose(cse_utils.nf_read_roi_zip(base_folder + 'regions/princeton_regions.zip', shape), [1, 2, 0])

    if os.path.exists(base_folder + 'regions/ben_regions.mat'):
        b = loadmat(base_folder + 'regions/ben_regions.mat')
        rois_2 = b['M']
    else:
        rois_2 = np.transpose(cse_utils.nf_read_roi_zip(base_folder + 'regions/ben_regions.zip', shape), [1, 2, 0])

    vmax_corr_perc = 95
    vmin_corr_perc = 5

    vmax_median_perc = 97
    vmin_median_perc = 5

    rois_1 = rois_1 * 1.
    rois_2 = rois_2 * 1.
    rois_1[rois_1 == 0] = np.nan
    rois_2[rois_2 == 0] = np.nan

    plot_images(img_corr, img_median, rois_1, rois_2, vmax_corr_perc, vmin_corr_perc, vmax_median_perc, vmin_median_perc)