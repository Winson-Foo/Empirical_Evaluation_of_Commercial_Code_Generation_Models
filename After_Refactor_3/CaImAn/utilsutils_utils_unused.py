#!/usr/bin/env python

import numpy as np

def get_patches_from_image(img: np.ndarray, shapes: np.ndarray, overlaps: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract patches from an image based on given shapes and overlaps.

    Args:
        img: The input image.
        shapes: Array specifying the shape of patches.
        overlaps: The number of overlapping pixels between patches.

    Returns:
        A tuple containing the extracted patches and their corresponding coordinates.
    """
    d1, d2 = img.shape
    rf = np.divide(shapes, 2)
    _, coords_2d = extract_patch_coordinates(d1, d2, rf=rf, stride=overlaps)
    imgs = np.empty(coords_2d.shape[:2], dtype=img.dtype)

    for idx_0, count_0 in enumerate(coords_2d):
        for idx_1, count_1 in enumerate(count_0):
            imgs[idx_0, idx_1] = img[count_1[0], count_1[1]]

    return imgs, coords_2d

def estimate_noise_mode(traces: np.ndarray, robust_std: bool = False, use_mode_fast: bool = False,
                        return_all: bool = False) -> np.ndarray:
    """
    Estimate the noise in the traces under the assumption that signals are sparse and only positive.
    The last dimension should be time.

    Args:
        traces: The input traces data.
        robust_std: If True, computes robust standard deviation using the interquartile range (IQR).
        use_mode_fast: If True, uses a fast mode estimation function.
        return_all: If True, returns the mode and the noise standard deviation.

    Returns:
        The estimated noise standard deviation or a tuple containing the mode and the noise standard deviation.
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
        Ns = np.round(np.sum(ff1 > 0, 1) * 0.5)
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