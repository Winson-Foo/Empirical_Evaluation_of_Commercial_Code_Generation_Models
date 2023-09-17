import pytest
import foolbox as fbn
import eagerpy as ep

def assert_shape_and_ndim(x, shape, ndim):
    assert x.shape == shape
    assert x.ndim == ndim

def assert_shape(x, shape):
    assert x.shape == shape

@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_atleast_kd_1d(dummy: ep.Tensor, k: int) -> None:
    x = ep.zeros(dummy, (10,))
    x = fbn.devutils.atleast_kd(x, k)
    assert_shape_and_ndim(x, (10,), k)

@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_atleast_kd_3d(dummy: ep.Tensor, k: int) -> None:
    x = ep.zeros(dummy, (10, 5, 3))
    x = fbn.devutils.atleast_kd(x, k)
    assert_shape_and_ndim(x, (10, 5, 3), max(k, 3))

def test_flatten_2d(dummy: ep.Tensor) -> None:
    x = ep.zeros(dummy, (4, 5))
    x = fbn.devutils.flatten(x)
    assert_shape(x, (4, 5))

def test_flatten_3d(dummy: ep.Tensor) -> None:
    x = ep.zeros(dummy, (4, 5, 6))
    x = fbn.devutils.flatten(x)
    assert_shape(x, (4, 30))

def test_flatten_4d(dummy: ep.Tensor) -> None:
    x = ep.zeros(dummy, (4, 5, 6, 7))
    x = fbn.devutils.flatten(x)
    assert_shape(x, (4, 210))