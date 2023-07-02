#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
mpl.use('TKAgg')
import sys
import psutil
import os
from ipyparallel import Client
import calblitz as cb
from glob import glob
import scipy.stats as st

def load_movie(folder_in, f_rate):
    fname_mov = os.path.join(os.path.split(folder_in)[0], os.path.split(folder_in)[-1] + 'MOV.hdf5')
    print(fname_mov)
    files = sorted(glob(os.path.join(os.path.split(folder_in)[0], 'images/*.tif')))
    print(files)

    m = cb.load_movie_chain(files, fr=f_rate)
    m.file_name = [os.path.basename(ttt) for ttt in m.file_name]
    m.save(fname_mov)
    del m

    return fname_mov

def create_labeling_images(f_name):
    cdir = os.path.dirname(f_name)

    print('loading')
    m = cb.load(f_name)

    print('corr image')
    img = m.local_correlations(eight_neighbours=True)
    im = cb.movie(img, fr=1)
    im.save(os.path.join(cdir, 'correlation_image.tif'))

    print('std image')
    img = np.std(m, 0)
    im = cb.movie(np.array(img), fr=1)
    im.save(os.path.join(cdir, 'std_projection.tif'))

    m1 = m.resize(1, 1, 1. / m.fr)

    print('median image')
    img = np.median(m1, 0)
    im = cb.movie(np.array(img), fr=1)
    im.save(os.path.join(cdir, 'median_projection.tif'))

    print('save BL')
    m1 = m1 - img
    m1.save(os.path.join(cdir, 'MOV_BL.tif'))
    m1 = m1.bilateral_blur_2D()
    m1.save(os.path.join(cdir, 'MOV_BL_BIL.tif'))
    m = np.array(m1)

    print('max image')
    img = np.max(m, 0)
    im = cb.movie(np.array(img), fr=1)
    im.save(os.path.join(cdir, 'max_projection.tif'))

    print('skew image')
    img = st.skew(m, 0)
    im = cb.movie(img, fr=1)
    im.save(os.path.join(cdir, 'skew_projection.tif'))

    del m
    del m1

    return f_name


def main():
    params = [
        ['/mnt/ceph/neuro/labeling/neurofinder.01.01/', 7.5],
    ]

    f_rates = np.array([el[1] for el in params])
    folders = np.array([el[0] for el in params])

    backend = 'local'
    if backend == 'SLURM':
        n_processes = int(os.environ.get('SLURM_NPROCS'))
    else:
        n_processes = np.maximum(int(psutil.cpu_count()), 1)

    print(('Using ' + str(n_processes) + ' processes'))

    single_thread = False

    if single_thread:
        dview = None
    else:
        try:
            c.close()
        except:
            print('C was not existing, creating one')

        print("Stopping  cluster to avoid unnecessary use of memory....")
        sys.stdout.flush()

        if backend == 'SLURM':
            try:
                cse.utilities.stop_server(is_slurm=True)
            except:
                print('Nothing to stop')

            slurm_script = '/mnt/xfs1/home/agiovann/SOFTWARE/Constrained_NMF/SLURM/slurmStart.sh'
            cse.utilities.start_server(slurm_script=slurm_script)

            pdir, profile = os.environ['IPPPDIR'], os.environ['IPPPROFILE']
            c = Client(ipython_dir=pdir, profile=profile)
        else:
            cse.utilities.stop_server()
            cse.utilities.start_server()
            c = Client()

        print(('Using ' + str(len(c)) + ' processes'))
        dview = c[:len(c)]

    pars = []
    for folder_in, f_rate in zip(folders, f_rates):
        print((folder_in, f_rate))
        pars.append([folder_in, f_rate])

    fls = c[:].map_sync(load_movie, pars)

    res = list(map(create_labeling_images, fls))

if __name__ == "__main__":
    main()