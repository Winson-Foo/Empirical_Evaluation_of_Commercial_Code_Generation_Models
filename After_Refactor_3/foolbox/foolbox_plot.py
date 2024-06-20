from typing import Tuple, Any, Optional
import numpy as np
import eagerpy as ep
import matplotlib.pyplot as plt

def _get_data_format(images: ep.Tensor, data_format: Optional[str]) -> Tuple[bool, bool]:
    channels_first = images.shape[1] == 1 or images.shape[1] == 3
    channels_last = images.shape[-1] == 1 or images.shape[-1] == 3
    if data_format is None:
        if channels_first == channels_last:
            raise ValueError("data_format ambiguous, please specify it explicitly")
    else:
        channels_first = data_format == "channels_first"
        channels_last = data_format == "channels_last"
        if not channels_first and not channels_last:
            raise ValueError("expected data_format to be 'channels_first' or 'channels_last'")
    return channels_first, channels_last

def _preprocess_images(images: ep.Tensor, channels_first: bool) -> np.ndarray:
    images = images.numpy()
    if channels_first:
        images = np.transpose(images, axes=(0, 2, 3, 1))
    return images

def _normalize_images(images: np.ndarray, bounds: Tuple[float, float]) -> np.ndarray:
    min_, max_ = bounds
    return (images - min_) / (max_ - min_)

def _create_figure(nrows: int, ncols: int, figsize: Tuple[float, float], **kwargs: Any) -> Tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(
        ncols=ncols,
        nrows=nrows,
        figsize=figsize,
        squeeze=False,
        constrained_layout=True,
        **kwargs,
    )
    return fig, axes

def show_images(images: ep.Tensor, n: Optional[int] = None, data_format: Optional[str] = None, 
                bounds: Tuple[float, float] = (0, 1), ncols: Optional[int] = None, 
                nrows: Optional[int] = None, figsize: Optional[Tuple[float, float]] = None,
                scale: float = 1, **kwargs: Any) -> None:
    x = ep.astensor(images)
    if x.ndim != 4:
        raise ValueError("expected images to have four dimensions: (N, C, H, W) or (N, H, W, C)")
    if n is not None:
        x = x[:n]
    channels_first, channels_last = _get_data_format(x, data_format)
    x = _preprocess_images(x, channels_first)
    x = _normalize_images(x, bounds)
    if nrows is None and ncols is None:
        nrows = 1
    if ncols is None:
        assert nrows is not None
        ncols = (len(x) + nrows - 1) // nrows
    elif nrows is None:
        nrows = (len(x) + ncols - 1) // ncols
    if figsize is None:
        figsize = (ncols * scale, nrows * scale)
    fig, axes = _create_figure(nrows, ncols, figsize, **kwargs)

    for row in range(nrows):
        for col in range(ncols):
            ax = axes[row][col]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis("off")
            i = row * ncols + col
            if i < len(x):
                if x.shape[-1] == 1:
                    ax.imshow(x[i][:, :, 0])
                else:
                    ax.imshow(x[i])