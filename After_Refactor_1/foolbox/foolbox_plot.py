from typing import Tuple, Any, Optional
import numpy as np
import eagerpy as ep
import matplotlib.pyplot as plt


def check_image_dimensions(images: ep.Tensor) -> None:
    if images.ndim != 4:
        raise ValueError("expected images to have four dimensions: (N, C, H, W) or (N, H, W, C)")


def check_data_format(data_format: Optional[str], x: ep.Tensor) -> Tuple[bool, bool]:
    if data_format is None:
        channels_first = x.shape[1] == 1 or x.shape[1] == 3
        channels_last = x.shape[-1] == 1 or x.shape[-1] == 3
        if channels_first == channels_last:
            raise ValueError("data_format ambiguous, please specify it explicitly")
    else:
        channels_first = data_format == "channels_first"
        channels_last = data_format == "channels_last" 
        if not channels_first and not channels_last:
            raise ValueError("expected data_format to be 'channels_first' or 'channels_last'")
    
    return channels_first, channels_last


def normalize_images(x: np.ndarray, bounds: Tuple[float, float]) -> np.ndarray:
    min_, max_ = bounds
    return (x - min_) / (max_ - min_)


def plot_images(x: np.ndarray, nrows: int, ncols: int, figsize: Tuple[float, float], **kwargs: Any) -> None:
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize, squeeze=False, constrained_layout=True, **kwargs)
    for ax, image in zip(axes.flatten(), x):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")
        if image.shape[-1] == 1:
            ax.imshow(image[:, :, 0])
        else:
            ax.imshow(image)


def images(
    images: Any,
    n: Optional[int] = None,
    data_format: Optional[str] = None,
    bounds: Tuple[float, float] = (0, 1),
    ncols: Optional[int] = None,
    nrows: Optional[int] = None,
    figsize: Optional[Tuple[float, float]] = None,
    scale: float = 1,
    **kwargs: Any,
) -> None:
    x: ep.Tensor = ep.astensor(images)
    check_image_dimensions(x)
    
    if n is not None:
        x = x[:n]
    
    channels_first, channels_last = check_data_format(data_format, x)
    x = x.numpy()
    
    if channels_first:
        x = np.transpose(x, axes=(0, 2, 3, 1))
    
    x = normalize_images(x, bounds)
    
    if nrows is None and ncols is None:
        nrows = 1
    elif ncols is None:
        ncols = (len(x) + nrows - 1) // nrows
    elif nrows is None:
        nrows = (len(x) + ncols - 1) // ncols
    
    if figsize is None:
        figsize = (ncols * scale, nrows * scale)
    
    plot_images(x, nrows, ncols, figsize, **kwargs)