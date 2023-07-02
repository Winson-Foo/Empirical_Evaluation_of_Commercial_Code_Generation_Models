from typing import Callable, Tuple
import eagerpy as ep

FuncType = Callable[..., Tuple[ep.Tensor, ep.Tensor]]
data: Dict[str, FuncType] = {}


def register(f: FuncType) -> FuncType:
    data[f.__name__] = f
    return f


@register
def example_4d(dummy: ep.Tensor) -> Tuple[ep.Tensor, ep.Tensor]:
    reference = ep.full(dummy, (10, 3, 32, 32), 0.2)
    perturbed = reference + 0.6
    return reference, perturbed


@register
def example_batch(dummy: ep.Tensor) -> Tuple[ep.Tensor, ep.Tensor]:
    x = ep.arange(dummy, 6).float32().reshape((2, 3))
    x = x / x.max()
    reference = x
    perturbed = 1 - x
    return reference, perturbed