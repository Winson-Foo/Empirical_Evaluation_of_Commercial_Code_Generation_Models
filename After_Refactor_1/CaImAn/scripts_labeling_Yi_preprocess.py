#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Wed Aug 10 12:35:08 2016
Updated on Sun Oct 24 15:00:00 2021
@author: agiovann
"""

from scipy.io import loadmat
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import calblitz as cb
import ca_source_extraction as cse


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


def load_image(base_folder, folder_name, file_name):
    img_path = os.path.join(base_folder, folder_name, file_name)
    return cb.load(img_path, fr=1)


def load_rois(base_folder, folder_name, file_name, shape):
    rois_path = os.path.join(base_folder, folder_name, file_name)
    if os.path.exists(rois_path):
        if file_name.endswith('.mat'):
            data = loadmat(rois_path)
            try:
                rois = data['allROIs']
            except:
                rois = data['M']
        elif file_name.endswith('.zip'):
            rois = np.transpose(cse.utilities.nf_read_roi_zip(rois_path, shape), [1, 2, 0])
    else:
        raise Exception(f"ROIs file does not exist at: {rois_path}")

    return rois


def plot_images(images, rois):
    num_plots = len(images)
    vmin_percentiles = [5, 5, 5]
    vmax_percentiles = [95, 97, 97]

    plt.figure(facecolor="white")
    plot_index = 1
    for i in range(num_plots):
        plt.subplot(2, 4, plot_index)
        plt.imshow(images[i], cmap='gray', vmax=np.percentile(images[i], vmax_percentiles[i]),
                   vmin=np.percentile(images[i], vmin_percentiles[i]))
        plt.imshow(np.nanmax(rois, -1), cmap='ocean', vmax=2, alpha=.2, vmin=0)
        plot_index += 1

    plt.subplot(2, 4, 4)
    plt.imshow(np.nanmax(rois, -1), cmap='ocean', vmax=2, alpha=.5, vmin=0)
    plt.imshow(np.nanmax(rois, -1), cmap='hot', alpha=.5, vmin=0, vmax=3)
    plt.axis('off')

    plt.subplot(2, 4, 8)
    plt.axis('off')
    plt.imshow(images[0], cmap='gray', vmax=np.percentile(images[0], vmax_percentiles[0]),
               vmin=np.percentile(images[0], vmin_percentiles[0]))
    plt.imshow(np.nanmax(rois, -1), cmap='ocean', vmax=2, alpha=.3, vmin=0)
    plt.imshow(np.nanmax(rois, -1), cmap='hot', alpha=.3, vmin=0, vmax=3)
    plt.axis('off')

    font = {'family': 'Myriad Pro',
            'weight': 'regular',
            'size': 30}
    plt.rc('font', **font)


def save_plot(base_folder):
    plt.rcParams['pdf.fonttype'] = 42
    plt.savefig(os.path.join(base_folder, 'comparison.pdf'))


if __name__ == '__main__':
    base_folder = '/mnt/ceph/neuro/labeling/k31_20151223_AM_150um_65mW_zoom2p2/'
    images_folder_name = 'projections'
    rois_folder_name = 'regions'

    img_corr = load_image(base_folder, images_folder_name, 'correlation_image.tif')
    img_median = load_image(base_folder, images_folder_name, 'median_projection.tif')
    shape = np.shape(img_median)

    try:
        rois_1 = load_rois(base_folder, rois_folder_name, 'princeton_regions.mat', shape)
    except Exception as e:
        print(f"Failed to load princeton_regions: {e}")
        A, C, template, idx_shapes, A_in = extract_sue_ann_info(os.path.join(base_folder, rois_folder_name, 'sue_ann_regions.mat'), base_folder)
        rois_1 = np.reshape(A.todense(), (shape[0], shape[1], -1), order='F')

    try:
        rois_2 = load_rois(base_folder, rois_folder_name, 'ben_regions.mat', shape)
    except Exception as e:
        print(f"Failed to load ben_regions: {e}")
        rois_2 = load_rois(base_folder, rois_folder_name, 'ben_regions.zip', shape)

    rois_1 = rois_1 * 1.
    rois_2 = rois_2 * 1.
    rois_1[np.isnan(rois_1)] = 0
    rois_2[np.isnan(rois_2)] = 0

    images = [img_corr, img_corr, img_corr, img_median, img_median, img_median]
    rois = [rois_1, rois_2, rois_1, rois_1, rois_2, rois_1]

    plot_images(images, rois)
    save_plot(base_folder)