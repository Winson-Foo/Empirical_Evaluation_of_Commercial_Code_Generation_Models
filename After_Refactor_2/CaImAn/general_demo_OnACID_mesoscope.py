#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

def download_files():
    fld_name = 'Mesoscope'
    fnames = []
    fnames.append(download_demo('Tolias_mesoscope_1.hdf5', fld_name))
    fnames.append(download_demo('Tolias_mesoscope_2.hdf5', fld_name))
    fnames.append(download_demo('Tolias_mesoscope_3.hdf5', fld_name))
    logging.info(fnames)
    return fnames

def set_parameters():
    fr = 15
    decay_time = 0.5
    gSig = (3, 3)
    p = 1
    min_SNR = 1
    ds_factor = 1
    gnb = 2
    gSig = tuple(np.ceil(np.array(gSig) / ds_factor).astype('int'))
    mot_corr = True
    pw_rigid = False
    max_shifts_online = np.ceil(10.).astype('int')
    sniper_mode = True
    rval_thr = 0.9
    init_batch = 200
    K = 2
    epochs = 1
    show_movie = False

    params_dict = {'fnames': fnames,
                   'fr': fr,
                   'decay_time': decay_time,
                   'gSig': gSig,
                   'p': p,
                   'min_SNR': min_SNR,
                   'rval_thr': rval_thr,
                   'ds_factor': ds_factor,
                   'nb': gnb,
                   'motion_correct': mot_corr,
                   'init_batch': init_batch,
                   'init_method': 'bare',
                   'normalize': True,
                   'sniper_mode': sniper_mode,
                   'K': K,
                   'epochs': epochs,
                   'max_shifts_online': max_shifts_online,
                   'pw_rigid': pw_rigid,
                   'dist_shape_update': True,
                   'min_num_trial': 10,
                   'show_movie': show_movie}
    opts = cnmf.params.CNMFParams(params_dict=params_dict)
    return opts

def fit_online(opts):
    cnm = cnmf.online_cnmf.OnACID(params=opts)
    cnm.fit_online()
    return cnm

def plot_contours(cnm):
    logging.info('Number of components: ' + str(cnm.estimates.A.shape[-1]))
    images = cm.load(fnames)
    Cn = images.local_correlations(swap_dim=False, frames_per_chunk=500)
    cnm.estimates.plot_contours(img=Cn, display_numbers=False)

def view_components(cnm):
    cnm.estimates.view_components(img=Cn)

def plot_timing_performance(cnm):
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

def save_results(cnm, fnames):
    c, dview, n_processes = \
        cm.cluster.setup_cluster(backend='local', n_processes=None,
                                 single_thread=False)
    if opts.online['motion_correct']:
        shifts = cnm.estimates.shifts[-cnm.estimates.C.shape[-1]:]
        if not opts.motion['pw_rigid']:
            memmap_file = cm.motion_correction.apply_shift_online(images, shifts,
                                                        save_base_name='MC')
        else:
            mc = cm.motion_correction.MotionCorrect(fnames, dview=dview,
                                                    **opts.get_group('motion'))

            mc.y_shifts_els = [[sx[0] for sx in sh] for sh in shifts]
            mc.x_shifts_els = [[sx[1] for sx in sh] for sh in shifts]
            memmap_file = mc.apply_shifts_movie(fnames, rigid_shifts=False,
                                                save_memmap=True,
                                                save_base_name='MC')
    else:
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
    fnames = download_files()
    opts = set_parameters()
    cnm = fit_online(opts)
    plot_contours(cnm)
    view_components(cnm)
    plot_timing_performance(cnm)
    save_results(cnm, fnames)

if __name__ == "__main__":
    main()