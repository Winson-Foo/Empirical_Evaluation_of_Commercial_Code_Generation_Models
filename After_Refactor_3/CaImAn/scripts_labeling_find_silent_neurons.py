#!/usr/bin/env python

import sys
import time
import numpy as np
from scipy.sparse import coo_matrix
import scipy
from matplotlib import pyplot as plt
import psutil
import glob
import os
from ca_source_extraction import utilities, pre_processing, spatial, temporal, plot_contours

def load_data(fname_new):
    Yr, dims, T = utilities.load_memmap(fname_new)
    Y = np.reshape(Yr, dims + (T,), order='F')
    return Y, Yr, dims, T

def preprocess_data(Yr, **options):
    Yr, sn, g, psx = pre_processing.preprocess_data(Yr, dview=None, **options['preprocess_params'])
    return Yr, sn, g, psx

def initialize_components(Y, normalize=True, **options):
    Atmp, Ctmp, b_in, f_in, center = initialization.initialize_components(Y, normalize=normalize, **options['init_params'])
    return Atmp, Ctmp, b_in, f_in, center

def update_spatial_components(Yr, C, f, A_in, **options):
    A, b, Cin, f = spatial.update_spatial_components(Yr, C=C, f=f, A_in=A_in.astype(bool), sn=sn, dview=None, **options['spatial_params'])
    return A, b, Cin, f

def update_temporal_components(Yr, A, b, C, f, **options):
    C, A, b, f, S, bl, c1, neurons_sn, g, YrA = temporal.update_temporal_components(Yr, A, b, C, f, dview=None, bl=None, c1=None, sn=None, g=None, **options['temporal_params'])
    return C, A, b, f, S, bl, c1, neurons_sn, g, YrA

def evaluate_components(traces, N, robust_std=True):
    idx_components, fitness, erfc = utilities.evaluate_components(traces, N=N, robust_std=robust_std)
    return idx_components, fitness, erfc

def extract_binary_masks_blob(A, min_radius, dims, minCircularity=0.5, minInertiaRatio=0.2, minConvexity=.8):
    masks_ws, pos_examples, neg_examples = utilities.extract_binary_masks_blob(A, min_radius, dims, minCircularity=minCircularity, minInertiaRatio=minInertiaRatio, minConvexity=minConvexity)
    return masks_ws, pos_examples, neg_examples

def view_patches_bar(Yr, A, C, b, f, dims, img, YrA=None):
    utilities.view_patches_bar(Yr, scipy.sparse.coo_matrix(A.tocsc()[:, idx_components]), C[idx_components, :], b, f, dims[0], dims[1], YrA=YrA[idx_components, :], img=Cn)
    
def main():
    try:
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
        print((1))
    except:
        print('Not launched under iPython')

    import matplotlib as mpl
    mpl.use('TKAgg')
    from time import time
    import tifffile
    import subprocess
    import pylab as pl
    from ipyparallel import Client
    
    # roughly number of cores on your machine minus 1
    n_processes = np.maximum(int(psutil.cpu_count()), 1)
    print(('using ' + str(n_processes) + ' processes'))
    
    # start cluster for efficient computation
    single_thread = False
    
    if single_thread:
        dview = None
    else:
        try:
            c.close()
        except:
            print('C was not existing, creating one')
        print("Stopping cluster to avoid unnecessary use of memory....")
        sys.stdout.flush()
        utilities.stop_server()
        utilities.start_server()
        c = Client()
        dview = c[:n_processes]
    
    base_folder = './'
    
    fname_new = 'Yr_d1_501_d2_398_d3_1_order_F_frames_369_.mmap'
    Y, Yr, dims, T = load_data(fname_new)
    
    K = Ain.shape[-1]  # number of neurons expected per patch
    gSig = [7, 7]  # expected half size of neurons
    merge_thresh = 1  # merging threshold, max correlation allowed
    p = 1  # order of the autoregressive system
    options = utilities.CNMFSetParms(Y, n_processes, p=p, gSig=gSig, K=K, ssub=2, tsub=2)
    
    t1 = time()
    Yr, sn, g, psx = preprocess_data(Yr, **options)
    print((time() - t1))

    A, C, b, f, center = initialize_components(Y, normalize=True, **options)

    pl.figure()
    crd = plot_contours(coo_matrix(Ain), Cn)
    pl.show()

    pl.figure()
    t1 = time()
    A, b, Cin, f = update_spatial_components(Yr, C=None, f=None, A_in=Ain.astype(bool), sn=sn, dview=None, **options)
    t_elSPATIAL = time() - t1
    print(t_elSPATIAL)
    pl.figure()
    crd = utilities.plot_contours(A, Cn)

    t1 = time()
    options['temporal_params']['p'] = 0
    C, A, b, f, S, bl, c1, neurons_sn, g, YrA = update_temporal_components(Yr, A, b, Cin, f, dview=dview, bl=None, c1=None, sn=None, g=None, **options)
    t_elTEMPORAL = time() - t1
    print(t_elTEMPORAL)

    traces = C + YrA
    traces = traces - scipy.signal.savgol_filter(traces, np.shape(traces)[1] / 2 * 2 - 1, 1, axis=1)

    idx_components, fitness, erfc = evaluate_components(traces, N=5, robust_std=True)
    idx_components = idx_components[fitness > -35]
    print((len(idx_components)))
    print((np.shape(A)))

    view_patches_bar(Yr, scipy.sparse.coo_matrix(A.tocsc()[:, idx_components]), C[idx_components, :], b, f, dims[0], dims[1], YrA=YrA[idx_components, :], img=Cn)

    pl.close()
    if not single_thread:
        c.close()
        utilities.stop_server()

    min_radius = 5  # min radius of expected blobs
    masks_ws, pos_examples, neg_examples = extract_binary_masks_blob(A.tocsc()[:, :], min_radius, dims, minCircularity=0.5, minInertiaRatio=0.2, minConvexity=.8)

    pl.subplot(1, 2, 1)

    final_masks = np.array(masks_ws)[pos_examples]
    pl.imshow(np.reshape(final_masks.max(0), dims, order='F'), vmax=1)
    pl.subplot(1, 2, 2)

    neg_examples_masks = np.array(masks_ws)[neg_examples]
    pl.imshow(np.reshape(neg_examples_masks.max(0), dims, order='F'), vmax=1)

    pl.imshow(np.reshape(A.tocsc()[:, neg_examples].mean(1), dims, order='F'))
    pl.imshow(np.reshape(A.tocsc()[:, pos_examples].mean(1), dims, order='F'))

    view_patches_bar(Yr, scipy.sparse.coo_matrix(A.tocsc()[:, pos_examples]), C[pos_examples, :], b, f, dims[0], dims[1], YrA=YrA[pos_examples, :], img=Cn)
    view_patches_bar(Yr, scipy.sparse.coo_matrix(A.tocsc()[:, neg_examples]), C[neg_examples, :], b, f, dims[0], dims[1], YrA=YrA[neg_examples, :], img=Cn)

if __name__ == "__main__":
    main()