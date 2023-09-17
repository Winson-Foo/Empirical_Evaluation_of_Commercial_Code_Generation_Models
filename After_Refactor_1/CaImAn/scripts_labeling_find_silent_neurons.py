#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.sparse import coo_matrix
from scipy.signal import savgol_filter
from skimage.measure import regionprops
from ipyparallel import Client

import ca_source_extraction as cse
from ca_source_extraction.pre_processing import preprocess_data
from ca_source_extraction.spatial import update_spatial_components
from ca_source_extraction.temporal import update_temporal_components
from ca_source_extraction.utilities import (
    load_memmap, local_correlations, nf_read_roi_zip, plot_contours, view_patches_bar, evaluate_components,
    extract_binary_masks_blob, create_fig, stop_server, start_server
)

# Load extensions if running in IPython
try:
    get_ipython().magic('load_ext autoreload')
    get_ipython().magic('autoreload 2')
    print((1))
except:
    print('Not launched under IPython')

import matplotlib as mpl
mpl.use('TKAgg')
# plt.ion()


# Define function to preprocess the data
def preprocess(input_data, **preprocess_params):
    Yr, sn, g, psx = preprocess_data(input_data, dview=None, **preprocess_params)
    return Yr, sn, g, psx


# Define function to update spatial components
def update_spatial(input_data, A_in, sn, **spatial_params):
    A, b, Cin, f = update_spatial_components(input_data, C=None, f=None, A_in=A_in.astype(bool), sn=sn, dview=None,
                                             **spatial_params)
    return A, b, Cin, f


# Define function to update temporal components
def update_temporal(input_data, A2, b2, Cin, f, **temporal_params):
    C, A, b, f, S, bl, c1, neurons_sn, g, YrA = update_temporal_components(input_data, A2, b2, Cin, f, dview=None, bl=None,
                                                      c1=None, sn=None, g=None, **temporal_params)
    return C, A, b, f, S, bl, c1, neurons_sn, g, YrA


# Define function to evaluate components and filter based on fitness
def filter_components(traces, N=5):
    idx_components, fitness, erfc = evaluate_components(traces, N=N, robust_std=True)
    idx_components = idx_components[fitness > -35]
    return idx_components


# Define function to extract binary masks based on blob detection
def extract_blob_masks(A, min_radius, dims):
    labels = []
    properties = []
    masks_ws = []
    pos_examples = []
    neg_examples = []

    for i, a in enumerate(A.T):
        img_a = np.reshape(a.toarray(), dims, order='F')
        blobs = img_a > 0
        labels_a = scipy.ndimage.label(blobs)[0]
        props_a = regionprops(labels_a)
        
        labels.append(labels_a)
        properties.append(props_a)

        for prop in props_a:
            if prop.area >= min_radius:
                pos_examples.append(len(masks_ws))
            else:
                neg_examples.append(len(masks_ws))

            masks_ws.append(labels_a == prop.label)
    
    return masks_ws, pos_examples, neg_examples


# Preprocessing parameters
preprocess_params = {
    'p': 1,
    'gSig': [7, 7],
    'K': K,
    'ssub': 2,
    'tsub': 2
}

# Spatial parameters
spatial_params = {
    'radius': 3,
    'only_init': False
}

# Temporal parameters
temporal_params = {
    'p': 1,
    'method': 'cvxpy',
    'low_rank': True,
    'nb': 2
}

# Load input data
input_data, dimensions, time_steps = load_memmap(fname_new)
Y = np.reshape(input_data, dimensions + (time_steps,), order='F')

# Compute local correlations
Cn = local_correlations(Y[:, :, :])
plt.imshow(Cn, cmap='gray')

# Load initial spatial components
rois_1 = np.transpose(nf_read_roi_zip(base_folder + 'regions/ben_regions.zip', np.shape(Cn)), [1, 2, 0])
initial_spatial_components = np.reshape(rois_1, [np.prod(np.shape(Cn)), -1], order='F')

# Number of neurons expected per patch
K = initial_spatial_components.shape[-1]

# Preprocess the data
Yr, sn, g, psx = preprocess(Y, **preprocess_params)

# Update spatial components
A, b, Cin, f = update_spatial(Yr, initial_spatial_components, sn, **spatial_params)
crd = plot_contours(A, Cn)

# Update temporal components
C, A, b, f, S, bl, c1, neurons_sn, g, YrA = update_temporal(Yr, A, b, Cin, f, **temporal_params)

# Update traces
traces = C + YrA
traces = traces - savgol_filter(traces, np.shape(traces)[1] // 2 * 2 - 1, 1, axis=1)

# Filter components
idx_components = filter_components(traces)

# View selected patches
view_patches_bar(Yr, scipy.sparse.coo_matrix(A.tocsc()[:, idx_components]), C[idx_components, :], b, f, dimensions[0], dimensions[1], YrA=YrA[idx_components, :], img=Cn)

# Refine spatial and temporal components
A2, b2, C2, f = update_spatial(Yr, C, f, A, sn=sn, **spatial_params)
C2, A2, b2, f2, S2, bl2, c12, neurons_sn2, g21, YrA = update_temporal(Yr, A2, b2, C2, f, **temporal_params)

# View patches after refinement
view_patches_bar(Yr, scipy.sparse.coo_matrix(A2.tocsc()[:, idx_components]), C2[idx_components, :], b2, f2, dimensions[0], dimensions[1], YrA=YrA[idx_components, :], img=Cn)

# Extract binary masks using blob detection
masks_ws, pos_examples, neg_examples = extract_blob_masks(A2.tocsc()[:, :], min_radius, dimensions)

plt.subplot(1, 2, 1)
final_masks = np.array(masks_ws)[pos_examples]
plt.imshow(np.reshape(final_masks.max(0), dimensions, order='F'), vmax=1)

plt.subplot(1, 2, 2)
neg_examples_masks = np.array(masks_ws)[neg_examples]
plt.imshow(np.reshape(neg_examples_masks.max(0), dimensions, order='F'), vmax=1)

# View patches of positive and negative examples
view_patches_bar(Yr, scipy.sparse.coo_matrix(A2.tocsc()[:, pos_examples]), C2[pos_examples, :], b2, f2, dimensions[0], dimensions[1], YrA=YrA[pos_examples, :], img=Cn)
view_patches_bar(Yr, scipy.sparse.coo_matrix(A2.tocsc()[:, neg_examples]), C2[neg_examples, :], b2, f2, dimensions[0], dimensions[1], YrA=YrA[neg_examples, :], img=Cn)