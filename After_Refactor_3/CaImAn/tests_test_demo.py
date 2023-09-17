#!/usr/bin/env python

import os
import numpy as np
import numpy.testing as npt
import caiman as cm
import caiman.paths
from caiman.source_extraction import cnmf


def load_movie_and_memory_map():
    fname_new = cm.save_memmap([os.path.join(caiman.paths.caiman_datadir(), 'example_movies', 'demoMovie.tif')],
                               base_name='Yr',
                               order='C')
    Yr, dims, T = cm.load_memmap(fname_new)
    return Yr, dims, T


def initialize_cnmf(n_processes, dview):
    return cnmf.CNMF(n_processes,
                     method_init='greedy_roi',
                     k=30,
                     gSig=[4, 4],
                     merge_thresh=.8,
                     p=2,
                     dview=dview,
                     Ain=None,
                     method_deconvolution='oasis')


def fit_cnmf(cnm, images):
    return cnm.fit(images)


def save_cnmf(cnm, filename):
    cnm.save(filename)


def load_cnmf(filename):
    return cnmf.cnmf.load_CNMF(filename)


def cleanup_dview(dview):
    try:
        dview.terminate()
    except:
        pass


def main(parallel=False):
    if parallel:
        c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)
    else:
        n_processes, dview = 2, None

    Yr, dims, T = load_movie_and_memory_map()
    cnm = initialize_cnmf(n_processes, dview)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    cnm = fit_cnmf(cnm, images)

    if parallel:
        cm.cluster.stop_server(dview=dview)

    save_cnmf(cnm, 'test_file.hdf5')
    cnm2 = load_cnmf('test_file.hdf5')

    npt.assert_allclose(cnm.estimates.A.sum(), cnm2.estimates.A.sum())
    npt.assert_allclose(cnm.estimates.C, cnm2.estimates.C)

    cleanup_dview(dview)


def test_single_thread():
    main()
    pass


def test_parallel():
    main(True)
    pass


if __name__ == "__main__":
    main()