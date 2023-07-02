#!/usr/bin/env python

import numpy as np

def get_patches_from_image(img: np.ndarray, shapes: np.ndarray, overlaps: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract patches from an image based on its shape and overlaps.

    Args:
    - img: Input image.
    - shapes: Shapes of the patches.
    - overlaps: Number of overlapping pixels between patches.

    Returns:
    - imgs: Extracted patches from the image.
    - coords_2d: Patch coordinates.

    """
    d1, d2 = img.shape
    rf = shapes / 2
    _, coords_2d = extract_patch_coordinates(d1, d2, rf=rf, stride=overlaps)
    imgs = img[coords_2d[:, :, 0], coords_2d[:, :, 1]]

    return imgs, coords_2d

def estimate_noise_mode(traces: np.ndarray, robust_std: bool = False, use_mode_fast: bool = False, return_all: bool = False) -> np.ndarray:
    """
    Estimate the noise in the traces under assumption that signals are sparse and only positive.
    The last dimension should be time.

    Args:
    - traces: Input traces.
    - robust_std (optional): Whether to compute the robust standard deviation.
    - use_mode_fast (optional): Whether to use a faster mode estimation method.
    - return_all (optional): Whether to return both the mode and standard deviation.

    Returns:
    - sd_r: Estimated noise standard deviation.

    """
    if use_mode_fast:
        md = mode_robust_fast(traces, axis=1)
    else:
        md = mode_robust(traces, axis=1)

    ff1 = traces - md[:, None]
    ff1 = -ff1 * (ff1 < 0)

    if robust_std:
        ff1 = np.sort(ff1, axis=1)
        ff1[ff1 == 0] = np.nan
        Ns = np.round(np.sum(ff1 > 0, 1) * .5)
        iqr_h = np.zeros(traces.shape[0])

        for idx, _ in enumerate(ff1):
            iqr_h[idx] = ff1[idx, -Ns[idx]]

        sd_r = 2 * iqr_h / 1.349
    else:
        Ns = np.sum(ff1 > 0, 1)
        sd_r = np.sqrt(np.sum(ff1**2, 1) / Ns)

    if return_all:
        return md, sd_r
    else:
        return sd_r