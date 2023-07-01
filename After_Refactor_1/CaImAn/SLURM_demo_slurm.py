#!/usr/bin/env python

import ca_source_extraction as cse
import glob
import matplotlib as mpl
mpl.use('TKAgg')
from matplotlib import pyplot as plt
import numpy as np
import os
import psutil
import pylab as pl
from scipy.sparse import coo_matrix
import shutil
import subprocess
import sys
from time import time
import tifffile
import time as tm

# Start the server
def start_server(ncpus=None, slurm_script=None):
    cse.utilities.start_server(ncpus=ncpus, slurm_script=slurm_script)

# Stop the server
def stop_server(is_slurm=True):
    cse.utilities.stop_server(is_slurm=is_slurm)

# Preprocess data and initialize components
def preprocess_data_and_initialize_components(Yr, options):
    Yr, sn, g = cse.pre_processing.preprocess_data(Yr, **options['preprocess_params'])
    Atmp, Ctmp, b_in, f_in, center = cse.initialization.initialize_components(
        Y, **options['init_params'])
    return Yr, sn, g, Atmp, Ctmp, b_in, f_in, center

# Refine components manually
def refine_components_manually(Y, gSig, Atmp, Ctmp, Cn):
    return cse.utilities.manually_refine_components(Y, gSig, coo_matrix(Atmp), Ctmp, Cn, thr=0.9)

# Update spatial components
def update_spatial_components(Yr, Cin, f_in, Ain, sn, options):
    return cse.spatial.update_spatial_components(Yr, Cin, f_in, Ain, sn=sn, **options['spatial_params'])

# Update temporal components
def update_temporal_components(Yr, A, b, Cin, f_in, bl, c1, sn, g, options):
    return cse.temporal.update_temporal_components(Yr, A, b, Cin, f_in, bl=None, c1=None, sn=None, g=None, **options['temporal_params'])

# Merge components
def merge_components(Yr, A, b, C, f, S, sn, temporal_params, spatial_params, bl=None, c1=None, thr=0.8, mx=50, fast_merge=True):
    return cse.merging.merge_components(Yr, A, b, C, f, S, sn, temporal_params, spatial_params, bl=bl, c1=c1, sn=neurons_sn, g=g, thr=0.8, mx=50, fast_merge=True)

# Order components
def order_components(A, C):
    return cse.utilities.order_components(A, C)

# Plot contours
def plot_contours(A, Cn, thr=0.9):
    return cse.utilities.plot_contours(A, Cn, thr=thr)

# View patches
def view_patches(Yr, A_or, C_or, b2, f2, d1, d2, YrA):
    return cse.utilities.view_patches(Yr, coo_matrix(A_or), C_or, b2, f2, d1, d2, YrA=YrA)

# View patches bar
def view_patches_bar(Yr, A_or, C_or, b2, f2, d1, d2, YrA):
    return cse.utilities.view_patches_bar(Yr, coo_matrix(A_or), C_or, b2, f2, d1, d2, YrA=YrA)

# Plot contours
def plot_contours(A_or, Cn, thr=0.9):
    plt.figure()
    return cse.utilities.plot_contours(A_or, Cn, thr=thr)

# Stop cluster
def stop_cluster():
    pl.close()
    cse.utilities.stop_server()

# Main function
def main():
    # Load movie
    filename = 'movies/demoMovie.tif'
    t = tifffile.TiffFile(filename)
    Yr = t.asarray().astype(dtype=np.float32)
    Yr = np.transpose(Yr, (1, 2, 0))
    d1, d2, T = Yr.shape
    Yr = np.reshape(Yr, (d1 * d2, T), order='F')
    np.save('Yr', Yr)
    Yr = np.load('Yr.npy', mmap_mode='r')
    Y = np.reshape(Yr, (d1, d2, T), order='F')
    Cn = cse.utilities.local_correlations(Y)

    # Set parameters
    n_processes = np.maximum(psutil.cpu_count() - 2, 1)
    p = 2
    options = cse.utilities.CNMFSetParms(Y, p=p, gSig=[4, 4], K=30)

    # Preprocess data and initialize components
    Yr, sn, g, Atmp, Ctmp, b_in, f_in, center = preprocess_data_and_initialize_components(Yr, options)

    # Refine components manually
    refine_components = False
    if refine_components:
        Ain, Cin = refine_components_manually(Y, options['init_params']['gSig'], coo_matrix(Atmp), Ctmp, Cn)
    else:
        Ain, Cin = Atmp, Ctmp

    # Plot estimated component
    crd = plot_contours(coo_matrix(Ain), Cn, thr=0.9)
    pl.show()

    # Update spatial components
    pl.close()
    A, b, Cin, f_in = update_spatial_components(Yr, Cin, f_in, Ain, sn, options)
    t_elSPATIAL = time() - t1
    print(t_elSPATIAL)
    plt.figure()
    crd = plot_contours(A, Cn, thr=0.9)

    # Update temporal components
    pl.close()
    t1 = time()
    options['temporal_params']['p'] = 0
    C, A, b, f, S, bl, c1, neurons_sn, g, YrA = update_temporal_components(
        Yr, A, b, Cin, f_in, bl=None, c1=None, sn=None, g=None, **options['temporal_params'])
    t_elTEMPORAL = time() - t1
    print(t_elTEMPORAL)

    # Merge components corresponding to the same neuron
    t1 = time()
    A_m, C_m, nr_m, merged_ROIs, S_m, bl_m, c1_m, sn_m, g_m = merge_components(
        Yr, A, b, C, f, S, sn, options['temporal_params'], options['spatial_params'], bl=bl, c1=c1, sn=neurons_sn, g=g, thr=0.8, mx=50, fast_merge=True)
    t_elMERGE = time() - t1
    print(t_elMERGE)

    # Plot merged components
    plt.figure()
    crd = plot_contours(A_m, Cn, thr=0.9)

    # Refine spatial and temporal components
    pl.close()
    t1 = time()
    A2, b2, C2, f = update_spatial_components(
        Yr, C_m, f, A_m, sn=sn, **options['spatial_params'])
    options['temporal_params']['p'] = p
    C2, A2, b2, f2, S2, bl2, c12, neurons_sn2, g21, YrA = update_temporal_components(
        Yr, A2, b2, C2, f, bl=None, c1=None, sn=None, g=None, **options['temporal_params'])
    print((time() - t1))

    # View ordered components
    A_or, C_or, srt = order_components(A2, C2)
    view_patches_bar(Yr, coo_matrix(A_or), C_or, b2, f2, d1, d2, YrA[srt, :])
    plt.show()

    # Plot ordered contours
    plt.figure()
    crd = plot_contours(A_or, Cn, thr=0.9)

    # Stop the cluster
    pl.close()
    stop_server()

if __name__ == "__main__":
    main()