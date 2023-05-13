import librosa
import librosa.display
import numpy as np
import random
import matplotlib
import torch
from .sparse_image_warp import sparse_image_warp

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def time_warp(spec, W=5):
    num_rows = spec.shape[1]
    spec_len = spec.shape[2]

    y = num_rows // 2
    horizontal_line_at_ctr = spec[0][y]

    point_to_warp = horizontal_line_at_ctr[random.randrange(W, spec_len - W)]

    dist_to_warp = random.randrange(-W, W)
    src_pts = torch.tensor([[[y, point_to_warp]]])
    dest_pts = torch.tensor([[[y, point_to_warp + dist_to_warp]]])
    try:
        warped_spectro, dense_flows = sparse_image_warp(spec, src_pts, dest_pts)
    except Exception as e:
        print("Error while warp image: ", e)
        return spec
    return warped_spectro.squeeze(3)


def spec_augment(mel_spectrogram, time_warping_para=40, frequency_masking_para=27,
                 time_masking_para=70, frequency_mask_num=1, time_mask_num=1):
    try:
        mel_spectrogram = mel_spectrogram.unsqueeze(0)
    except Exception as e:
        print("Error while unsqueeze the spectrogram: ", e)
        return None
    v = mel_spectrogram.shape[1]
    tau = mel_spectrogram.shape[2]

    # Step 1 : Time warping
    warped_mel_spectrogram = time_warp(mel_spectrogram)

    # Step 2 : Frequency masking
    for i in range(frequency_mask_num):
        f = np.random.uniform(low=0.0, high=frequency_masking_para)
        f = int(f)
        if v - f < 0:
            continue
        f0 = random.randint(0, v - f)
        try:
            warped_mel_spectrogram[:, f0:f0 + f, :] = 0
        except Exception as e:
            print("Error while applying frequency mask: ", e)

    # Step 3 : Time masking
    for i in range(time_mask_num):
        t = np.random.uniform(low=0.0, high=time_masking_para)
        t = int(t)
        if tau - t < 0:
            continue
        t0 = random.randint(0, tau - t)
        try:
            warped_mel_spectrogram[:, :, t0:t0 + t] = 0
        except Exception as e:
            print("Error while applying time mask: ", e)

    return warped_mel_spectrogram.squeeze()


def visualization_spectrogram(mel_spectrogram, title):
    try:
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(librosa.power_to_db(mel_spectrogram[0, :, :], ref=np.max), y_axis='mel', fmax=8000,
                                 x_axis='time')
        plt.title(title)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("Error while visualizing the spectrogram: ", e)