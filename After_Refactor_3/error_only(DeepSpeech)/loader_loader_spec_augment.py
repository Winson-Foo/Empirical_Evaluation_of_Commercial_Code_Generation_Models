import librosa
import librosa.display
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .sparse_image_warp import sparse_image_warp
import torch


def time_warp(spec, W=5):
    try:
        num_rows = spec.shape[1]
        spec_len = spec.shape[2]
    except AttributeError:
        raise ValueError("Invalid Spec - Expecting a 3D numpy array")

    y = num_rows // 2
    horizontal_line_at_ctr = spec[0][y]

    point_to_warp = horizontal_line_at_ctr[random.randrange(W, spec_len-W)]

    dist_to_warp = random.randrange(-W, W)
    src_pts = torch.tensor([[[y, point_to_warp]]])
    dest_pts = torch.tensor([[[y, point_to_warp + dist_to_warp]]])
    try:
        warped_spectro, dense_flows = sparse_image_warp(spec, src_pts, dest_pts)
    except Exception as e:
        raise ValueError("Error occured during time warping - {}".format(str(e)))

    return warped_spectro.squeeze(3)


def spec_augment(mel_spectrogram, time_warping_para=40, frequency_masking_para=27,
                 time_masking_para=70, frequency_mask_num=1, time_mask_num=1):
    try:
        mel_spectrogram = mel_spectrogram.unsqueeze(0)
    except AttributeError:
        raise ValueError("Invalid Mel Spectrogram - Expecting a 2D numpy array")

    v = mel_spectrogram.shape[1]
    tau = mel_spectrogram.shape[2]

    # Step 1 : Time warping
    try:
        warped_mel_spectrogram = time_warp(mel_spectrogram, W=time_warping_para)
    except ValueError as e:
        raise ValueError("Error in time warping - {}".format(str(e)))

    # Step 2 : Frequency masking
    for i in range(frequency_mask_num):
        f = np.random.uniform(low=0.0, high=frequency_masking_para)
        f = int(f)
        if v - f < 0:
            continue
        f0 = random.randint(0, v-f)
        warped_mel_spectrogram[:, f0:f0+f, :] = 0

    # Step 3 : Time masking
    for i in range(time_mask_num):
        t = np.random.uniform(low=0.0, high=time_masking_para)
        t = int(t)
        if tau - t < 0:
            continue
        t0 = random.randint(0, tau-t)
        warped_mel_spectrogram[:, :, t0:t0+t] = 0

    return warped_mel_spectrogram.squeeze(0)


def visualization_spectrogram(mel_spectrogram, title):
    """visualizing result of SpecAugment
    # Arguments:
      mel_spectrogram(ndarray): mel_spectrogram to visualize.
      title(String): plot figure's title
    """
    # Show mel-spectrogram using librosa's specshow.
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(mel_spectrogram[0, :, :], ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
    # plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()