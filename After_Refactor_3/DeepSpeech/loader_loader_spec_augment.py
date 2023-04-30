# spec_augment.py

import numpy as np
import random
import matplotlib.pyplot as plt

from .sparse_image_warp import sparse_image_warp
from typing import Tuple

def time_warp(spec: np.ndarray, warp_param: float=5) -> np.ndarray:
    """
    Applies time warping to a mel spectrogram.
    
    Args:
    - spec: The mel spectrogram to be warped.
    - warp_param: Augmentation parameter, "time warp parameter W".
        If none, default = 5.
        
    Returns:
    - The warped mel spectrogram.
    """
    
    num_rows = spec.shape[1]
    spec_len = spec.shape[2]

    y = num_rows // 2
    horizontal_line_at_ctr = spec[0][y]

    point_to_warp = horizontal_line_at_ctr[
        random.randrange(warp_param, spec_len - warp_param)]
    
    dist_to_warp = random.randrange(-warp_param, warp_param)
    
    src_pts = torch.tensor([[[y, point_to_warp]]])
    dest_pts = torch.tensor([[[y, point_to_warp + dist_to_warp]]])
    
    warped_spectro, dense_flows = sparse_image_warp(spec, src_pts, dest_pts)

    return warped_spectro.squeeze(3)


def frequency_mask(spec: np.ndarray,
                   masking_lines: int=1,
                   mask_param: float=27) -> np.ndarray:
    """
    Applies frequency masking to a mel spectrogram.
    
    Args:
    - spec: The mel spectrogram to be masked.
    - masking_lines: The number of masking lines to apply.
    - mask_param: Augmentation parameter, "frequency mask parameter F"
        If none, default = 27.
        
    Returns:
    - The masked mel spectrogram.
    """
    
    for i in range(masking_lines):
        
        f = np.random.uniform(low=0.0, high=mask_param)
        f = int(f)
        
        if spec.shape[1] - f < 0:
            continue
        
        f0 = random.randint(0, spec.shape[1] - f)
        
        spec[:, f0:f0+f, :] = 0
    
    return spec


def time_mask(spec: np.ndarray,
              masking_lines: int=1,
              mask_param: float=70) -> np.ndarray:
    """
    Applies time masking to a mel spectrogram.
    
    Args:
    - spec: The mel spectrogram to be masked.
    - masking_lines: The number of masking lines to apply.
    - mask_param: Augmentation parameter, "time mask parameter T"
        If none, default = 70.
        
    Returns:
    - The masked mel spectrogram.
    """
    
    for i in range(masking_lines):
        
        t = np.random.uniform(low=0.0, high=mask_param)
        t = int(t)
        
        if spec.shape[2] - t < 0:
            continue
        
        t0 = random.randint(0, spec.shape[2] - t)
        
        spec[:, :, t0:t0+t] = 0
    
    return spec


def spec_augment(mel_spectrogram: np.ndarray,
                 time_warping_para: float=40,
                 frequency_masking_para: float=27,
                 time_masking_para: float=70,
                 frequency_mask_num: int=1,
                 time_mask_num: int=1) -> np.ndarray:
    """
    Applies SpecAugment to a mel spectrogram.
    
    Args:
    - mel_spectrogram: The mel spectrogram to be augmented.
    - time_warping_para: Augmentation parameter, "time warp parameter W".
        If none, default = 40.
    - frequency_masking_para: Augmentation parameter, "frequency mask parameter F"
        If none, default = 27.
    - time_masking_para: Augmentation parameter, "time mask parameter T"
        If none, default = 70.
    - frequency_mask_num: The number of frequency masking lines to apply.
    - time_mask_num: The number of time masking lines to apply.
    
    Returns:
    - The augmented mel spectrogram.
    """
    
    mel_spectrogram = mel_spectrogram.unsqueeze(0)

    # Step 1 : Time warping
    warped_mel_spectrogram = time_warp(mel_spectrogram, time_warping_para)

    # Step 2 : Frequency masking
    warped_mel_spectrogram = frequency_mask(warped_mel_spectrogram,
                                             frequency_mask_num,
                                             frequency_masking_para)

    # Step 3 : Time masking
    warped_mel_spectrogram = time_mask(warped_mel_spectrogram,
                                        time_mask_num,
                                        time_masking_para)

    return warped_mel_spectrogram.squeeze()


def visualize_mel_spectrogram(mel_spectrogram: np.ndarray, title: str) -> None:
    """Visualizes a mel spectrogram.
    
    Args:
    - mel_spectrogram: The mel spectrogram to be visualized.
    - title: The title of the plot.
        
    """
    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(mel_spectrogram[0, :, :], ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
    plt.title(title)
    plt.tight_layout()
    plt.show()