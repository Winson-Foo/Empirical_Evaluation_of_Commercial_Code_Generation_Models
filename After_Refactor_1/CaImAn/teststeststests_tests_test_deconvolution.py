#!/usr/bin/env python

import logging
import numpy.testing as npt
import numpy as np
from time import time

from caiman.source_extraction.cnmf.deconvolution import constrained_foopsi

logging.basicConfig(
    format="%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s] [%(process)d] %(message)s",
    level=logging.DEBUG
)


def generate_traces(impulse_response=[.95], noise_std=.2, duration=1000, framerate=30, firing_rate=.5, baseline=10, num_traces=1, seed=0):
    """
    Generate data from homogeneous Poisson Process
    
    :param impulse_response: array, shape (p,), default=[.95]
        Parameter(s) of the AR(p) process that models the fluorescence impulse response.
    :param noise_std: float, default .2
        Noise standard deviation.
    :param duration: int, default 1000
        Duration.
    :param framerate: int, default 30
        Frame rate.
    :param firing_rate: int, default .5
        Neural firing rate.
    :param baseline: int, default 10
        Baseline.
    :param num_traces: int, default 1
        Number of generated traces.
    :param seed: int, default 0
        Seed of random number generator.
        
    :return: 
        - noisy_traces: array, shape (duration,)
            Noisy fluorescence data.
        - calcium_traces: array, shape (duration,)
            Calcium traces (without sn).
        - spike_trains: array, shape (duration,)
            Spike trains.
    """
    
    np.random.seed(seed)
    num_frames = int(framerate * duration)
    noisy_traces = np.zeros((num_traces, num_frames))
    calcium_traces = np.random.rand(num_traces, num_frames) < firing_rate / float(framerate)
    spike_trains = calcium_traces.astype(float)
    for i in range(2, num_frames):
        if len(impulse_response) == 2:
            spike_trains[:, i] += impulse_response[0] * spike_trains[:, i - 1] + impulse_response[1] * spike_trains[:, i - 2]
        else:
            spike_trains[:, i] += impulse_response[0] * spike_trains[:, i - 1]
    noisy_traces = baseline + spike_trains + noise_std * np.random.randn(num_traces, num_frames)
    return noisy_traces, calcium_traces, spike_trains


def run_deconvolution(method, p):
    start_time = time()
    impulse_response = np.array([[.95], [1.7, -.71]][p - 1])
    for i, noise_std in enumerate([.2, .5]):  # high and low SNR
        noisy_traces, calcium_traces, spike_trains = [a[0] for a in generate_traces(impulse_response, noise_std)]
        result = constrained_foopsi(noisy_traces, g=impulse_response, sn=noise_std, p=p, method=method)
        npt.assert_allclose(np.corrcoef(result[0], calcium_traces)[0, 1], 1, [.01, .1][i])
        npt.assert_allclose(np.corrcoef(result[-2], spike_trains)[0, 1], 1, [.03, .3][i])
    logging.debug(['\n', ''][p - 1] + ' %5s AR%d   %.4fs' % (method, p, time() - start_time))


def test_oasis():
    run_deconvolution('oasis', 1)
    run_deconvolution('oasis', 2)