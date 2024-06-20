import numpy as np
import pytest

from distances import distances
from data import data, FuncType

FuncType = Callable[..., Tuple[ep.Tensor, ep.Tensor]]


@pytest.fixture(scope="session", params=list(data.keys()))
def reference_perturbed(request: Any, dummy: ep.Tensor) -> Tuple[ep.Tensor, ep.Tensor]:
    return data[request.param](dummy)


@pytest.mark.parametrize("p", [0, 1, 2, ep.inf])
def test_distance(reference_perturbed: Tuple[ep.Tensor, ep.Tensor], p: float) -> None:
    reference, perturbed = reference_perturbed

    actual = distances[p](reference, perturbed).numpy()

    diff = perturbed.numpy() - reference.numpy()
    diff = diff.reshape((len(diff), -1))
    desired = np.linalg.norm(diff, ord=p, axis=-1)

    np.testing.assert_allclose(actual, desired, rtol=1e-5)


@pytest.mark.parametrize("p", [0, 1, 2, ep.inf])
def test_distance_repr_str(p: float) -> None:
    assert str(p) in repr(distances[p])
    assert str(p) in str(distances[p])


@pytest.mark.parametrize("p", [0, 1, 2, ep.inf])
def test_distance_clip(
    reference_perturbed: Tuple[ep.Tensor, ep.Tensor], p: float
) -> None:
    reference, perturbed = reference_perturbed

    ds = distances[p](reference, perturbed).numpy()
    epsilon = np.median(ds)
    too_large = ds > epsilon

    desired = np.where(too_large, epsilon, ds)

    perturbed = distances[p].clip_perturbation(reference, perturbed, epsilon)
    actual = distances[p](reference, perturbed).numpy()

    np.testing.assert_allclose(actual, desired)