import eagerpy as ep
import numpy as np
import pytest

def rescale_images(images: ep.Tensor, shape: tuple[int, int, int, int], axis: int) -> ep.Tensor:
    return ep.resize(images, shape, axis)

def test_rescale_axis(request, dummy):
    backend = request.config.option.backend
    if backend == "numpy":
        pytest.skip()

    input_shape = (16, 3, 64, 64)
    target_shape = (16, 3, 128, 128)

    input_images_np = np.random.uniform(0.0, 1.0, size=input_shape)
    input_images_ep = ep.astensor(input_images_np)
    target_images_ep = rescale_images(input_images_ep, target_shape, 1)
    target_images_np = target_images_ep.numpy()

    input_images = ep.from_numpy(dummy, input_images_np)
    input_images_ep = ep.astensor(input_images)
    target_images_ep = rescale_images(input_images_ep, target_shape, 1)
    target_images = target_images_ep.numpy()

    assert np.allclose(target_images_np, target_images, atol=1e-5)

def test_rescale_axis_nhwc(request, dummy):
    backend = request.config.option.backend
    if backend == "numpy":
        pytest.skip()

    input_shape = (16, 64, 64, 3)
    target_shape = (16, 128, 128, 3)

    input_images_np = np.random.uniform(0.0, 1.0, size=input_shape)
    input_images_ep = ep.astensor(input_images_np)
    target_images_ep = rescale_images(input_images_ep, target_shape, -1)
    target_images_np = target_images_ep.numpy()

    input_images = ep.from_numpy(dummy, input_images_np)
    input_images_ep = ep.astensor(input_images)
    target_images_ep = rescale_images(input_images_ep, target_shape, -1)
    target_images = target_images_ep.numpy()

    assert np.allclose(target_images_np, target_images, atol=1e-5)