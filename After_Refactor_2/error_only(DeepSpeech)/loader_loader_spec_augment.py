import librosa
import librosa.display
import numpy as np
import random
import matplotlib.pyplot as plt
from .sparse_image_warp import sparse_image_warp
import torch

def time_warp(spec, W=5):
    num_rows = spec.shape[1]
    spec_len = spec.shape[2]
    y = num_rows // 2
    horizontal_line_at_ctr = spec[0][y]
    if len(horizontal_line_at_ctr) != spec_len:
        raise ValueError('Invalid shape of spectrogram')
    point_to_warp = horizontal_line_at_ctr[random.randrange(W, spec_len-W)]
    if not isinstance(point_to_warp, torch.Tensor):
        point_to_warp = torch.tensor(point_to_warp)
    dist_to_warp = random.randrange(-W, W)
    src_pts = torch.tensor([[[y, point_to_warp]]])
    dest_pts = torch.tensor([[[y, point_to_warp + dist_to_warp]]])
    warped_spectro, dense_flows = sparse_image_warp(spec, src_pts, dest_pts)
    return warped_spectro.squeeze(3)


def spec_augment(mel_spectrogram, time_warping_para=40, frequency_masking_para=27,
                 time_masking_para=70, frequency_mask_num=1, time_mask_num=1):
    mel_spectrogram = torch.tensor(mel_spectrogram).unsqueeze(0)
    v = mel_spectrogram.shape[1]
    tau = mel_spectrogram.shape[2]
    if frequency_masking_para > v:
        raise ValueError('frequency_masking_para should be <= number of mel bands')
    if time_masking_para > tau:
        raise ValueError('time_masking_para should be <= number of timesteps')
    # Step 1 : Time warping
    warped_mel_spectrogram = time_warp(mel_spectrogram, time_warping_para)
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
    return warped_mel_spectrogram.squeeze().numpy()


def visualization_spectrogram(mel_spectrogram, title):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(mel_spectrogram, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
    plt.title(title)
    plt.tight_layout()
    plt.show()