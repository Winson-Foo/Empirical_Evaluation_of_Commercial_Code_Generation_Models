import os
import numpy.testing as npt
from caiman.source_extraction import cnmf
from caiman.paths import caiman_datadir


def initialize_parameters():
    params_dict = {
        'framerate': 10,  # frame rate (Hz)
        'fnames': [os.path.join(caiman_datadir(), 'example_movies', 'demoMovie.tif')],
        'decay_time': .75,  # approximate length of transient event in seconds
        'gSig': [6, 6],  # expected half size of neurons
        'p': 1,  # order of AR indicator dynamics
        'min_SNR': 1,  # minimum SNR for accepting candidate components
        'thresh_CNN_noisy': 0.65,  # CNN threshold for candidate components
        'nb': 2,  # number of background components
        'init_method': 'cnmf',  # initialization method
        'init_batch': 400,  # number of frames for initialization
        'rf': 16,  # patch size
        'stride': 3,  # amount of overlap between patches
        'sniper_mode': True,
        'K': 4  # max number of components in each patch
    }
    return cnmf.params.CNMFParams(params_dict=params_dict)


def run_cnmf(params):
    cnm = cnmf.online_cnmf.OnACID(params=params)
    cnm.fit_online()
    cnm.save('test_online.hdf5')
    cnm2 = cnmf.online_cnmf.load_OnlineCNMF('test_online.hdf5')
    npt.assert_allclose(cnm.estimates.A.sum(), cnm2.estimates.A.sum())
    npt.assert_allclose(cnm.estimates.C, cnm2.estimates.C)


def demo():
    params = initialize_parameters()
    run_cnmf(params)


def test_onacid():
    demo()
    pass