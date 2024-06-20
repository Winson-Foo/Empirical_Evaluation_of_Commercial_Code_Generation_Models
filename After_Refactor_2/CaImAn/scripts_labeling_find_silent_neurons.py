#!/usr/bin/env python

import glob
import os
import sys
import time

import numpy as np
import psutil
import scipy
import scipy.signal
import scipy.sparse
from ipyparallel import Client
from matplotlib import pyplot as plt
import matplotlib as mpl
import tifffile

import ca_source_extraction as cse
from ca_source_extraction.pre_processing import preprocess_data
from ca_source_extraction.spatial import (update_spatial_components,
                                          plot_contours)
from ca_source_extraction.temporal import update_temporal_components
from ca_source_extraction.utilities import (load_memmap,
                                            local_correlations,
                                            nf_read_roi_zip,
                                            CNMFSetParms,
                                            view_patches_bar,
                                            stop_server, start_server,
                                            evaluate_components,
                                            extract_binary_masks_blob)

# Set matplotlib backend
mpl.use('TKAgg')


def main():
    try:
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
        print(1)
    except:
        print('Not launched under iPython')

    n_processes = max(int(psutil.cpu_count()), 1)
    print(f'Using {n_processes} processes')

    single_thread = False

    if single_thread:
        dview = None
    else:
        try:
            c.close()
        except:
            print('C was not existing, creating one')
        print("Stopping cluster to avoid unnecessary use of memory...")
        sys.stdout.flush()
        stop_server()
        start_server()
        c = Client()
        dview = c[:n_processes]

    # Load memmap
    base_folder = './'
    fname_new = 'Yr_d1_501_d2_398_d3_1_order_F_frames_369_.mmap'
    Yr, dims, T = load_memmap(fname_new)
    Y = np.reshape(Yr, dims + (T,), order='F')

    # Calculate local correlations
    Cn = local_correlations(Y[:, :, :])
    plt.imshow(Cn, cmap='gray')

    # Load ROIs
    rois_1 = np.transpose(nf_read_roi_zip(base_folder +
                                          'regions/ben_regions.zip', np.shape(Cn)), [1, 2, 0])
    Ain = np.reshape(rois_1, [np.prod(np.shape(Cn)), -1], order='F')

    K = Ain.shape[-1]
    gSig = [7, 7]
    p = 1
    options = CNMFSetParms(Y, n_processes, p=p, gSig=gSig, K=K, ssub=2, tsub=2)

    # Preprocess data and initialize components
    t1 = time.time()
    Yr, _, _, _ = preprocess_data(Yr, dview=dview, **options['preprocess_params'])
    print(time.time() - t1)

    # Update spatial components
    plt.figure()
    t1 = time.time()
    A, b, Cin, f = update_spatial_components(Yr, C=None, f=None, A_in=Ain.astype(bool), sn=None, dview=None, **options['spatial_params'])
    t_elSPATIAL = time.time() - t1
    print(t_elSPATIAL)
    plt.figure()
    crd = plot_contours(A, Cn)

    # Update temporal components
    t1 = time.time()
    options['temporal_params']['p'] = 0
    C, A, b, f, S, _, _, neurons_sn, _, YrA = update_temporal_components(Yr, A, b, Cin, f, dview=dview, sn=None, **options['temporal_params'])
    t_elTEMPORAL = time.time() - t1
    print(t_elTEMPORAL)

    traces = C + YrA
    traces = traces - scipy.signal.savgol_filter(traces, np.shape(traces)[1] // 2 * 2 - 1, 1, axis=1)

    idx_components, fitness, _ = evaluate_components(traces, N=5, robust_std=True)
    idx_components = idx_components[fitness > -35]
    print(len(idx_components))
    print(np.shape(A))

    # Visualize components
    view_patches_bar(Yr, scipy.sparse.coo_matrix(A.tocsc()[:, idx_components]), C[:, idx_components], b, f, dims[0], dims[1], YrA=YrA[idx_components, :], img=Cn)


    # Refine spatial and temporal components
    plt.close()
    t1 = time.time()
    A2, b2, C2, f = update_spatial_components(Yr, C, f, A, sn=None, dview=dview, **options['spatial_params'])
    options['temporal_params']['p'] = p
    C2, A2, b2, f2, S2, bl2, c12, neurons_sn2, _, YrA = update_temporal_components(
        Yr, A2, b2, C2, f, dview=dview, sn=None, **options['temporal_params'])
    print(time.time() - t1)

    plt.figure()
    crd = plot_contours(A2, Cn)

    traces = C2 + YrA
    traces = traces - scipy.signal.savgol_filter(traces, np.shape(traces)[
                                               1] // 2 * 2 - 1, 1, axis=1)
    idx_components, fitness, _ = evaluate_components(
        traces, N=5, robust_std=True)
    idx_components = idx_components[fitness < -15]

    view_patches_bar(Yr, scipy.sparse.coo_matrix(A2.tocsc()[:, idx_components]), C2[:, idx_components], b2, f2, dims[0], dims[1], YrA=YrA[idx_components, :], img=Cn)

    plt.figure()
    plt.imshow(np.reshape(A2.tocsc()[:, neg_examples].mean(1), dims, order='F'))

    plt.figure()
    plt.imshow(np.reshape(A2.tocsc()[:, pos_examples].mean(1), dims, order='F'))

    # Select blobs
    min_radius = 5
    masks_ws, pos_examples, neg_examples = extract_binary_masks_blob(A2.tocsc()[:, :], min_radius, dims, minCircularity=0.5, minInertiaRatio=0.2, minConvexity=.8)

    plt.subplot(1, 2, 1)
    final_masks = np.array(masks_ws)[pos_examples]
    plt.imshow(np.reshape(final_masks.max(0), dims, order='F'), vmax=1)
    plt.subplot(1, 2, 2)

    neg_examples_masks = np.array(masks_ws)[neg_examples]
    plt.imshow(np.reshape(neg_examples_masks.max(0), dims, order='F'), vmax=1)

    view_patches_bar(Yr, scipy.sparse.coo_matrix(A2.tocsc()[:, pos_examples]), C2[pos_examples, :], b2, f2, dims[0], dims[1], YrA=YrA[pos_examples, :], img=Cn)

    view_patches_bar(Yr, scipy.sparse.coo_matrix(A2.tocsc()[:, neg_examples]), C2[neg_examples, :], b2, f2, dims[0], dims[1], YrA=YrA[neg_examples, :], img=Cn)

    plt.show()

    if not single_thread:
        c.close()
        stop_server()

if __name__ == "__main__":
    main()