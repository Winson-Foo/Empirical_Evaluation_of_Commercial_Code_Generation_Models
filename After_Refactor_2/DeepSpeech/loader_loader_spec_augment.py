import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from .sparse_image_warp import sparse_image_warp

SUPPORTED_POLICIES = {
    "None": (0, 0, -1, 0, -1, -1),
    "LB": (80, 27,  1, 100, 1.0, 1),
    "LD": (80, 27, 2, 100, 1.0, 2),
    "SM": (40, 15, 2,  70, 0.2, 2),
    "SS": (40, 27, 2,  70, 0.2, 2)
}

SUPPORTED_DATASETS = {
    "LB": "LibriSpeech basic",
    "LD": "LibriSpeech double",
    "SM": "Switchboard mild",
    "SS": "Switchboard strong"
}

def apply_time_warp(spec, warp_para):
    num_rows = spec.shape[1]
    spec_len = spec.shape[2]

    y = num_rows // 2
    horizontal_line_at_ctr = spec[0][y]
    # assert len(horizontal_line_at_ctr) == spec_len

    point_to_warp = horizontal_line_at_ctr[random.randrange(warp_para, spec_len - warp_para)]
    # assert isinstance(point_to_warp, torch.Tensor)

    # Uniform distribution from (0,W) with chance to be up to W negative
    dist_to_warp = random.randrange(-warp_para, warp_para)
    src_pts = torch.tensor([[[y, point_to_warp]]])
    dest_pts = torch.tensor([[[y, point_to_warp + dist_to_warp]]])
    warped_spectro, dense_flows = sparse_image_warp(spec, src_pts, dest_pts)

    return warped_spectro.squeeze(3)

def apply_frequency_masking(mel_spec, F, m_F):
    v = mel_spec.shape[1]
    mask = np.ones((mel_spec.shape[0], v, mel_spec.shape[-1]))
    for i in range(m_F):
        f = np.random.uniform(low=0.0, high=F)
        f = int(f)
        if v - f < 0:
            continue
        f0 = random.randint(0, v-f)
        mask[:, f0:f0+f, :] = 0
    return mel_spec * mask

def apply_time_masking(mel_spec, T, m_T):
    tau = mel_spec.shape[-1]
    # Step 3 : Time masking
    mask = np.ones((mel_spec.shape[0], mel_spec.shape[1], tau))
    for i in range(m_T):
        t = np.random.uniform(low=0.0, high=T)
        t = int(t)
        if tau - t < 0:
            continue
        t0 = random.randint(0, tau - t)
        mask[:, :, t0:t0+t] = 0
    return mel_spec * mask

def apply_spec_augment(mel_spectrogram, policy):
    """Apply Spec augmentation.
    # Arguments:
      mel_spectrogram(numpy array): audio file path of you want to warping and masking.
      policy(string): name of policy to use from SUPPORTED_POLICIES.
    # Returns
      mel_spectrogram(numpy array): warped and masked mel spectrogram.
    """
    mel_spectrogram = mel_spectrogram.squeeze(0)
    warp_para, F, m_F, T, p, m_T = SUPPORTED_POLICIES[policy]

    # Step 1 : Time warping
    warped_mel_spectrogram = apply_time_warp(mel_spectrogram, warp_para)

    # Step 2 : Frequency masking
    warped_mel_spectrogram = apply_frequency_masking(warped_mel_spectrogram, F, m_F)

    # Step 3 : Time masking
    warped_mel_spectrogram = apply_time_masking(warped_mel_spectrogram, T, m_T)

    return warped_mel_spectrogram

def visualize_spectrogram(mel_spectrogram, title):
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