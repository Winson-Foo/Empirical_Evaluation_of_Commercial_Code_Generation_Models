#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import pandas as pd
import pylab as pl

import cv2
try:
    cv2.setNumThreads(1)
except:
    print('OpenCV is naturally single threaded')

try:
    if __IPYTHON__:
        print((1))
        # this is used for debugging purposes only. allows to reload classes when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    print('Not launched under iPython')

import caiman as cm
from caiman.base.rois import nf_read_roi_zip
from caiman.source_extraction.cnmf import cnmf as cnmf

def compare_labelers_consensus(folders):
    for folder_out in folders:
        projection_img_median = os.path.join(folder_out, 'projections/median_projection.tif')
        projection_img_correlation = os.path.join(folder_out, 'projections/correlation_image.tif')
        folder_in = os.path.join(folder_out, 'regions')
        
        performance_all = dict()
        fls = list(glob.glob(os.path.join(folder_in, '*_nd.zip')))
        consensus_counter = dict()
        fl1 = os.path.join(folder_in, 'joined_consensus_active_regions.zip')
        
        for fl2 in fls:
            print([fl1, fl2])
            Cn = cm.load(projection_img_correlation)
            shape = Cn.shape

            roi_1, names_1 = nf_read_roi_zip(fl1, shape, return_names=True)
            roi_2, names_2 = nf_read_roi_zip(fl2, shape, return_names=True)
            
            lab1, lab2 = fl1.split('/')[-1][:-4], fl2.split('/')[-1][:-4]

            tp_gt, tp_comp, fn_gt, fp_comp, performance = cm.base.rois.nf_match_neurons_in_binary_masks(roi_1, roi_2, thresh_cost=.7, min_dist=10,
                                                                                      print_assignment=False, plot_results=False, Cn=Cn, labels=[lab1, lab2])

            performance['tp_gt'] = tp_gt
            performance['tp_comp'] = tp_comp
            performance['fn_gt'] = fn_gt
            performance['fp_comp'] = fp_comp

            performance_all[fl1, fl2] = performance

        np.savez(os.path.join(folder_in, 'comparison_labelers_consensus.npz'), performance_all=performance_all)


def process_folder_out(folders):
    f1s = []
    names = []
    
    for folder_out in folders:
        projection_img_median = os.path.join(folder_out, 'projections/median_projection.tif')
        projection_img_correlation = os.path.join(folder_out, 'projections/correlation_image.tif')
        folder_in = os.path.join(folder_out, 'regions')
        
        print('********' + folder_out)
        with np.load(os.path.join(folder_in, 'comparison_labelers_consensus.npz'), encoding='latin1') as ld:
            pf = ld['performance_all'][()]
            
            for key in pf:
                if pf[key]['f1_score'] <= 1:
                    print(str(pf[key]['f1_score'])[:5] + ':' + str(key[-1]).split('/')[-1].split('_')[0])
                    f1s.append(pf[key]['f1_score'])
                    names.append(str(key[-1]).split('/')[-1].split('_')[0])
    
    df = pd.DataFrame({'names': names, 'f1s': f1s})
    g = df['f1s'].groupby(df['names'])
    
    return g


def plot_results(folders):    
    counter = 1
    
    for folder_out in folders:
        projection_img_median = os.path.join(folder_out, 'projections/median_projection.tif')
        projection_img_correlation = os.path.join(folder_out, 'projections/correlation_image.tif')
        folder_in = os.path.join(folder_out, 'regions')
        
        with np.load(os.path.join(folder_in, 'comparison_labelers.npz')) as ld:
            pl.figure(figsize=(20, 10))
            counter += 1
            count = 0
            img = cm.load(projection_img_correlation)
            img[np.isnan(img)] = np.nanmean(img)
            
            for fl_m in [os.path.join(folder_in, '*_matches.zip'), os.path.join(folder_in, '*1_mismatches.zip'), os.path.join(folder_in, '*0_mismatches.zip')]:
                count += 1
                pl.subplot(1, 3, count)
                vmin, vmax = np.percentile(img, (5, 95))
                pl.imshow(img, vmin=vmin, vmax=vmax)
                rois_m = nf_read_roi_zip(glob.glob(fl_m)[0], img.shape)
                rois_m = rois_m.sum(0) * img.max()
                rois_m[rois_m == 0] = np.nan
                pl.imshow(rois_m, alpha=.5, cmap='hot', vmin=vmin, vmax=vmax)
                pl.axis('off')
                pl.xlabel(os.path.basename(fl_m))
                
            pl.title(os.path.basename(folder_out))
            pl.pause(1)

# Main code

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

compare_labelers_consensus(folders_out)

g = process_folder_out(folders_out)

plot_results(folders_out)