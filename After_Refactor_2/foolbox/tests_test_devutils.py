import pytest
import foolbox as fbn
import eagerpy as ep


def flatten_tensor(x: ep.Tensor, num_dims: int) -> ep.Tensor:
    x = fbn.devutils.flatten(x)
    target_shape = (x.shape[0],) + (x.shape[1] * x.shape[2],) * (num_dims - 2)
    assert x.shape == target_shape
    return x


def test_atleast_kd_1d(dummy: ep.Tensor, k: int) -> None:
    x = ep.zeros(dummy, (10,))
    x = fbn.devutils.atleast_kd(x, k)
    assert x.shape[0] == 10
    assert x.ndim == k


def test_atleast_kd_3d(dummy: ep.Tensor, k: int) -> None:
    x = ep.zeros(dummy, (10, 5, 3))
    x = fbn.devutils.atleast_kd(x, k)
    assert x.shape[:3] == (10, 5, 3)
    assert x.ndim == max(k, 3)


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_flatten_2d(dummy: ep.Tensor, k: int) -> None:
    x = ep.zeros(dummy, (4, 5))
    x = flatten_tensor(x, k)
    assert x.shape == (4, 5)


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_flatten_3d(dummy: ep.Tensor, k: int) -> None:
    x = ep.zeros(dummy, (4, 5, 6))
    x = flatten_tensor(x, k)
    assert x.shape == (4, 30)


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_flatten_4d(dummy: ep.Tensor, k: int) -> None:
    x = ep.zeros(dummy, (4, 5, 6, 7))
    x = flatten_tensor(x, k)
    assert x.shape == (4, 210)