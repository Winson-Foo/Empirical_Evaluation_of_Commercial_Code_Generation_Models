from typing import Union, Any
import eagerpy as ep

from ..models import Model
from ..criteria import Misclassification
from ..distances import l2
from ..devutils import flatten, atleast_kd
from .base import FixedEpsilonAttack, get_criterion, raise_if_kwargs, verify_input_bounds

class VirtualAdversarialAttack(FixedEpsilonAttack):
    """
    Second-order gradient-based attack on the logits.
    """

    distance = l2

    def __init__(self, steps: int, xi: float = 1e-6):
        self.steps = steps
        self.xi = xi

    def run(self, model: Model, inputs: ep.Tensor, criterion: Union[Misclassification, ep.Tensor], *,
            epsilon: float, **kwargs: Any) -> ep.Tensor:
        raise_if_kwargs(kwargs)
        x, restore_type = ep.astensor_(inputs)
        criterion_ = get_criterion(criterion)
        del inputs, criterion, kwargs

        verify_input_bounds(x, model)

        N = len(x)

        if isinstance(criterion_, Misclassification):
            classes = criterion_.labels
        else:
            raise ValueError("Unsupported criterion")

        if classes.shape != (N,):
            raise ValueError(f"Expected labels to have shape ({N},), got {classes.shape}")

        bounds = model.bounds

        def loss_fun(delta: ep.Tensor, logits: ep.Tensor) -> ep.Tensor:
            assert x.shape[0] == logits.shape[0]
            assert delta.shape == x.shape

            x_hat = x + delta
            logits_hat = model(x_hat)
            loss = ep.kl_div_with_logits(logits, logits_hat).sum()

            return loss

        value_and_grad = ep.value_and_grad_fn(x, loss_fun, has_aux=False)

        clean_logits = model(x)

        d = ep.normal(x, shape=x.shape, mean=0, stddev=1)
        for it in range(self.steps):
            d = d * self.xi / atleast_kd(ep.norms.l2(flatten(d), axis=-1), x.ndim)
            _, grad = value_and_grad(d, clean_logits)
            d = grad
            d = (bounds[1] - bounds[0]) * d

            if ep.any(ep.norms.l2(flatten(d), axis=-1) < 1e-64):
                raise RuntimeError("Gradient vanished; this can happen if xi is too small.")

        final_delta = epsilon / atleast_kd(ep.norms.l2(flatten(d), axis=-1), d.ndim) * d
        x_adv = ep.clip(x + final_delta, *bounds)
        
        return restore_type(x_adv)