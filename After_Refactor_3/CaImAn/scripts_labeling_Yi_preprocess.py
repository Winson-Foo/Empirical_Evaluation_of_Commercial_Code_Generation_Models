#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Wed Aug 10 12:35:08 2016

@author: agiovann
"""

import numpy as np
from PIL import Image
from scipy.io import loadmat
import matplotlib.pyplot as plt
import os
import ca_source_extraction as cse
import calblitz as cb


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


def load_image(file_path, percentile_low, percentile_high):
    img = cb.load(file_path, fr=1)
    percentile_low_val = np.percentile(img, percentile_low)
    percentile_high_val = np.percentile(img, percentile_high)
    return img, percentile_low_val, percentile_high_val


def load_roi(file_path, shape):
    if os.path.exists(file_path):
        data = loadmat(file_path)
        try:
            return data['allROIs']
        except:
            return data['M']
    else:
        return np.transpose(cse.utilities.nf_read_roi_zip(file_path, shape), [1, 2, 0])


def preprocess_roi(rois):
    rois = rois * 1.
    rois[rois == 0] = np.nan
    return rois


def plot_images(img, rois_1, rois_2, cmap, alpha, vmin, vmax):
    plt.imshow(img, cmap=cmap, vmax=vmax, vmin=vmin)
    plt.imshow(np.nanmax(rois_1, -1), cmap='ocean', vmax=2, alpha=alpha, vmin=0)
    plt.imshow(np.nanmax(rois_2, -1), cmap='hot', alpha=alpha, vmin=0, vmax=3)
    plt.axis('off')


def save_figure(file_path):
    plt.savefig(file_path)


def main():
    base_folder = '/mnt/ceph/neuro/labeling/k31_20151223_AM_150um_65mW_zoom2p2/'
    img_corr_file = base_folder + 'projections/correlation_image.tif'
    img_median_file = base_folder + 'projections/median_projection.tif'
    rois_princeton_file = base_folder + 'regions/princeton_regions.mat'
    rois_sue_ann_file = base_folder + 'regions/sue_ann_regions.mat'
    rois_ben_file = base_folder + 'regions/ben_regions.mat'
    rois_princeton_zip = base_folder + 'regions/princeton_regions.zip'
    rois_ben_zip = base_folder + 'regions/ben_regions.zip'
    comparison_file = base_folder + 'comparison.pdf'

    img_corr, vmin_corr_perc, vmax_corr_perc = load_image(
        img_corr_file, 5, 95)
    img_median, vmin_median_perc, vmax_median_perc = load_image(
        img_median_file, 5, 97)

    shape = np.shape(img_median)

    rois_1 = load_roi(rois_princeton_file, shape)
    if rois_1 is None:
        rois_1 = load_roi(rois_sue_ann_file, shape)
    if rois_1 is None:
        rois_1 = np.transpose(cse.utilities.nf_read_roi_zip(
            rois_princeton_zip, shape), [1, 2, 0])

    rois_2 = load_roi(rois_ben_file, shape)
    if rois_2 is None:
        rois_2 = np.transpose(cse.utilities.nf_read_roi_zip(
            rois_ben_zip, shape), [1, 2, 0])

    rois_1 = preprocess_roi(rois_1)
    rois_2 = preprocess_roi(rois_2)

    plt.figure(facecolor="white")
    plt.subplot(2, 4, 1)
    plot_images(img_corr, rois_1, rois_2, 'gray', 0.2, vmin_corr_perc, vmax_corr_perc)
    plt.ylabel('CORR IMAGE')
    plt.title('PRINCETON')
    plt.axis('off')
    plt.subplot(2, 4, 2)
    plot_images(img_corr, rois_2, rois_2, 'gray', 0.2, vmin_corr_perc, vmax_corr_perc)
    plt.title('BEN')
    plt.axis('off')
    plt.subplot(2, 4, 3)
    plot_images(img_corr, np.zeros_like(rois_1), np.zeros_like(rois_2), 'gray', 0, vmin_corr_perc, vmax_corr_perc)
    plt.axis('off')
    plt.subplot(2, 4, 5)
    plot_images(img_median, rois_1, rois_2, 'gray', 0.2, vmin_median_perc, vmax_median_perc)
    plt.ylabel('MEDIAN')
    plt.axis('off')
    plt.subplot(2, 4, 6)
    plot_images(img_median, rois_2, rois_2, 'gray', 0.2, vmin_median_perc, vmax_median_perc)
    plt.axis('off')
    plt.subplot(2, 4, 7)
    plot_images(img_median, np.zeros_like(rois_1), np.zeros_like(rois_2), 'gray', 0, vmin_median_perc, vmax_median_perc)
    plt.axis('off')
    plt.subplot(2, 4, 4)

    plt.imshow(np.nanmax(rois_1, -1), cmap='ocean', vmax=2, alpha=.5, vmin=0)
    plt.imshow(np.nanmax(rois_2, -1), cmap='hot', alpha=.5, vmin=0, vmax=3)
    plt.axis('off')

    plt.subplot(2, 4, 8)
    plt.axis('off')
    plot_images(img_corr, rois_1, rois_2, 'gray', 0.3, vmin_corr_perc, vmax_corr_perc)
    plt.axis('off')

    font = {'family': 'Myriad Pro',
            'weight': 'regular',
            'size': 30}

    plt.rc('font', **font)

    plt.rcParams['pdf.fonttype'] = 42
    save_figure(comparison_file)


if __name__ == '__main__':
    main()