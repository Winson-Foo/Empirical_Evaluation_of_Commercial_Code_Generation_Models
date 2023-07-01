#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Complete pipeline for online processing using CaImAn Online (OnACID).
The demo demonstates the analysis of a sequence of files using the CaImAn online
algorithm. The steps include i) motion correction, ii) tracking current 
components, iii) detecting new components, iv) updating of spatial footprints.
The script demonstrates how to construct and use the params and online_cnmf
objects required for the analysis, and presents the various parameters that
can be passed as options. A plot of the processing time for the various steps
of the algorithm is also included.
@author: Eftychios Pnevmatikakis @epnev
Special thanks to Andreas Tolias and his lab at Baylor College of Medicine
for sharing the data used in this demo.
"""

import glob
import numpy as np
import os
import logging
import matplotlib.pyplot as plt

import caiman as cm
from caiman.paths import caiman_datadir
from caiman.source_extraction import cnmf as cnmf
from caiman.utils.utils import download_demo

logging.basicConfig(format=
                    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]"\
                    "[%(process)d] %(message)s",
                    level=logging.INFO)

def process_files(fnames):
    show_movie = False # show the movie as the data gets processed

    params_dict = {
        'fnames': fnames,
        'fr': 15,
        'decay_time': 0.5,
        'gSig': (3, 3),
        'p': 1,
        'min_SNR': 1,
        'rval_thr': 0.9,
        'ds_factor': 1,
        'nb': 2,
        'motion_correct': True,
        'init_batch': 200,
        'init_method': 'bare',
        'normalize': True,
        'sniper_mode': True,
        'K': 2,
        'epochs': 1,
        'max_shifts_online': 10,
        'pw_rigid': False,
        'dist_shape_update': True,
        'min_num_trial': 10,
        'show_movie': show_movie
    }
    opts = cnmf.params.CNMFParams(params_dict=params_dict)

    cnm = cnmf.online_cnmf.OnACID(params=opts)
    cnm.fit_online()

    logging.info('Number of components: ' + str(cnm.estimates.A.shape[-1]))
    images = cm.load(fnames)
    Cn = images.local_correlations(swap_dim=False, frames_per_chunk=500)
    cnm.estimates.plot_contours(img=Cn, display_numbers=False)
    cnm.estimates.view_components(img=Cn)

    T_motion = 1e3*np.array(cnm.t_motion)
    T_detect = 1e3*np.array(cnm.t_detect)
    T_shapes = 1e3*np.array(cnm.t_shapes)
    T_track = 1e3*np.array(cnm.t_online) - T_motion - T_detect - T_shapes
    plt.figure()
    plt.stackplot(np.arange(len(T_motion)), T_motion, T_track, T_detect, T_shapes)
    plt.legend(labels=['motion', 'tracking', 'detect', 'shapes'], loc=2)
    plt.title('Processing time allocation')
    plt.xlabel('Frame #')
    plt.ylabel('Processing time [ms]')

    c, dview, n_processes = \
        cm.cluster.setup_cluster(backend='local', n_processes=None,
                                single_thread=False)

    if opts.online['motion_correct']:
        shifts = cnm.estimates.shifts[-cnm.estimates.C.shape[-1]:]
        if not opts.motion['pw_rigid']:
            memmap_file = cm.motion_correction.apply_shift_online(
                images, shifts, save_base_name='MC'
            )
        else:
            mc = cm.motion_correction.MotionCorrect(fnames, dview=dview,
                                                    **opts.get_group('motion'))

            mc.y_shifts_els = [[sx[0] for sx in sh] for sh in shifts]
            mc.x_shifts_els = [[sx[1] for sx in sh] for sh in shifts]
            memmap_file = mc.apply_shifts_movie(
                fnames, rigid_shifts=False,
                save_memmap=True, save_base_name='MC'
            )
    else:  # To do: apply non-rigid shifts on the fly
        memmap_file = images.save(fnames[0][:-4] + 'mmap')
    cnm.mmap_file = memmap_file
    Yr, dims, T = cm.load_memmap(memmap_file)

    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    min_SNR = 2
    rval_thr = 0.85
    use_cnn = True
    min_cnn_thr = 0.99
    cnn_lowest = 0.1

    cnm.params.set('quality',   {'min_SNR': min_SNR,
                                'rval_thr': rval_thr,
                                'use_cnn': use_cnn,
                                'min_cnn_thr': min_cnn_thr,
                                'cnn_lowest': cnn_lowest})

    cnm.estimates.evaluate_components(images, cnm.params, dview=dview)
    cnm.estimates.Cn = Cn
    cnm.save(os.path.splitext(fnames[0])[0]+'_results.hdf5')

    dview.terminate()

def main():
    # folder inside ./example_movies where files will be saved
    fld_name = 'Mesoscope'
    fnames = []
    fnames.append(download_demo('Tolias_mesoscope_1.hdf5', fld_name))
    fnames.append(download_demo('Tolias_mesoscope_2.hdf5', fld_name))
    fnames.append(download_demo('Tolias_mesoscope_3.hdf5', fld_name))

    # your list of files should look something like this
    logging.info(fnames)

    process_files(fnames)

if __name__ == "__main__":
    main()