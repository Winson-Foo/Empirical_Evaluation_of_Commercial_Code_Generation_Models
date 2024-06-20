import eagerpy as ep
import numpy as np
import pytest
from typing import Any


def setup(request: Any) -> Any:
    backend = request.config.option.backend
    if backend == "numpy":
        pytest.skip()

    dummy = ep.astensor(np.zeros((1, 1, 1, 1)))
    return dummy


def rescale_images_axis(x: ep.Tensor, shape: tuple[int, int, int, int], axis: int) -> ep.Tensor:
    return ep.resize(x, shape, axis=axis)


def rescale_images_nhwc(x: ep.Tensor, shape: tuple[int, int, int, int], axis: int) -> ep.Tensor:
    if axis == -1:
        x = x.transpose((0, 3, 1, 2))
    x = ep.resize(x, shape)
    if axis == -1:
        x = x.transpose((0, 2, 3, 1))
    return x


def test_rescale_axis(request: Any) -> None:
    dummy = setup(request)

    x_np = np.random.uniform(0.0, 1.0, size=(16, 3, 64, 64))

    x = ep.astensor(x_np)
    x_up = rescale_images_axis(x, (16, 3, 128, 128), 1)
    x_up_np = x_up.numpy()

    assert np.allclose(x_up_np, x_np, atol=1e-5)


def test_rescale_axis_nhwc(request: Any) -> None:
    dummy = setup(request)

    x_np = np.random.uniform(0.0, 1.0, size=(16, 64, 64, 3))

    x = ep.astensor(x_np)
    x_up = rescale_images_nhwc(x, (16, 128, 128, 3), -1)
    x_up_np = x_up.numpy()

    assert np.allclose(x_up_np, x_np, atol=1e-5)