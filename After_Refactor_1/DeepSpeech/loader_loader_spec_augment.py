import librosa
import librosa.display
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .sparse_image_warp import sparse_image_warp
import torch


POLICIES = {'None': {'W': 0, 'F': 0, 'm_F': '-', 'T': 0, 'p': '-', 'm_T': '-'},
            'LB': {'W': 80, 'F': 27, 'm_F': 1, 'T': 100, 'p': 1.0, 'm_T': 1},
            'LD': {'W': 80, 'F': 27, 'm_F': 2, 'T': 100, 'p': 1.0, 'm_T': 2},
            'SM': {'W': 40, 'F': 15, 'm_F': 2, 'T': 70, 'p': 0.2, 'm_T': 2},
            'SS': {'W': 40, 'F': 27, 'm_F': 2, 'T': 70, 'p': 0.2, 'm_T': 2}}

PARAMETERS = {'time_warping_para': 40, 'frequency_masking_para': 27,
              'time_masking_para': 70, 'frequency_mask_num': 1, 'time_mask_num': 1}


def time_warp(input_mel):
    """
    Performs time warping on the input mel-spectrogram using Tensorflow's image_sparse_warp function.

    Args:
        input_mel (ndarray): The input mel-spectrogram.

    Returns:
        ndarray: The time warped mel-spectrogram.

    """
    num_rows = input_mel.shape[1]
    spec_len = input_mel.shape[2]

    y = num_rows // 2
    horizontal_line_at_ctr = input_mel[0][y]

    point_to_warp = horizontal_line_at_ctr[random.randrange(POLICIES['LB']['W'], spec_len-POLICIES['LB']['W'])]
    dist_to_warp = random.randrange(-POLICIES['LB']['W'], POLICIES['LB']['W'])
    src_pts = torch.tensor([[[y, point_to_warp]]])
    dest_pts = torch.tensor([[[y, point_to_warp + dist_to_warp]]])
    warped_spectro, dense_flows = sparse_image_warp(input_mel, src_pts, dest_pts)

    return warped_spectro.squeeze(3)


def freq_masking(input_mel, num_masks, mask_param):
    """
    Performs frequency masking on the input mel-spectrogram.

    Args:
        input_mel (ndarray): The input mel-spectrogram.
        num_masks (int): The number of frequency masking lines to apply.
        mask_param (int): The maximum number of frequency bins to mask.

    Returns:
        ndarray: The frequency masked mel-spectrogram.

    """
    num_rows = input_mel.shape[1]
    for i in range(num_masks):
        f = np.random.uniform(low=0.0, high=mask_param)
        f = int(f)
        if num_rows - f < 0:
            continue
        f0 = random.randint(0, num_rows-f)
        input_mel[:, f0:f0+f, :] = 0

    return input_mel


def time_masking(input_mel, num_masks, mask_param):
    """
    Performs time masking on the input mel-spectrogram.

    Args:
        input_mel (ndarray): The input mel-spectrogram.
        num_masks (int): The number of time masking lines to apply.
        mask_param (int): The maximum number of time frames to mask.

    Returns:
        ndarray: The time masked mel-spectrogram.

    """
    spec_len = input_mel.shape[2]
    for i in range(num_masks):
        t = np.random.uniform(low=0.0, high=mask_param)
        t = int(t)
        if spec_len - t < 0:
            continue
        t0 = random.randint(0, spec_len-t)
        input_mel[:, :, t0:t0+t] = 0

    return input_mel


def spec_augment(input_mel, params=PARAMETERS, policy='LB'):
    """
    Performs SpecAugment on the input mel-spectrogram.

    Args:
        input_mel (ndarray): The input mel-spectrogram.
        params (dict): The parameters for SpecAugment (default: PARAMETERS).
        policy (str): The policy to use for SpecAugment (default: 'LB').

    Returns:
        ndarray: The augmented mel-spectrogram.

    """
    if input_mel.ndim == 2:
        input_mel = np.expand_dims(input_mel, axis=0)

    assert input_mel.ndim == 3, "Input mel-spectrogram has incorrect shape."

    W = POLICIES[policy]['W']
    F = POLICIES[policy]['F']
    m_F = POLICIES[policy]['m_F']
    T = POLICIES[policy]['T']
    p = POLICIES[policy]['p']
    m_T = POLICIES[policy]['m_T']

    # Step 1: Time warping
    warped_mel_spectrogram = time_warp(input_mel)

    # Step 2: Frequency masking
    freq_masked_mel_spectrogram = freq_masking(warped_mel_spectrogram, m_F, F)

    # Step 3: Time masking
    augmented_mel_spectrogram = time_masking(freq_masked_mel_spectrogram, m_T, T)

    return augmented_mel_spectrogram.squeeze()


def visualization_spectrogram(mel_spectrogram, title):
    """
    Visualizes the mel-spectrogram using librosa's specshow function.

    Args:
        mel_spectrogram (ndarray): The mel-spectrogram to visualize.
        title (str): The title for the visualization.

    """
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(mel_spectrogram[0, :, :], ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
    plt.title(title)
    plt.tight_layout()
    plt.show()