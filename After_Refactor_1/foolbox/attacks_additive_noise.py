from typing import Union, Any, cast
from abc import ABC, abstractmethod
import eagerpy as ep
from ..devutils import flatten, atleast_kd
from ..distances import l2, linf
from .base import FixedEpsilonAttack, Criterion, Model, T, get_criterion, get_is_adversarial, raise_if_kwargs, verify_input_bounds

class BaseAdditiveNoiseAttack(FixedEpsilonAttack, ABC):
    def run(self, model: Model, inputs: T, criterion: Union[Criterion, Any] = None, *, epsilon: float, **kwargs: Any) -> T:
        raise_if_kwargs(kwargs)
        x, restore_type = ep.astensor_(inputs)
        del inputs, criterion, kwargs
        verify_input_bounds(x, model)
        min_bound, max_bound = model.bounds
        p = self.sample_noise(x)
        epsilons = self.get_epsilons(x, p, epsilon, min_bound, max_bound)
        x = x + epsilons * p
        x = x.clip(min_bound, max_bound)
        return restore_type(x)

    @abstractmethod
    def sample_noise(self, x: ep.Tensor) -> ep.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_epsilons(self, x: ep.Tensor, p: ep.Tensor, epsilon: float, min_bound: float, max_bound: float) -> ep.Tensor:
        raise NotImplementedError


class L2Mixin:
    distance = l2

    def get_epsilons(self, x: ep.Tensor, p: ep.Tensor, epsilon: float, min_bound: float, max_bound: float) -> ep.Tensor:
        norms = flatten(p).norms.l2(axis=-1)
        return epsilon / atleast_kd(norms, p.ndim)


class L2ClippingAwareMixin:
    distance = l2

    def get_epsilons(self, x: ep.Tensor, p: ep.Tensor, epsilon: float, min_bound: float, max_bound: float) -> ep.Tensor:
        return cast(ep.Tensor, l2_clipping_aware_rescaling(x, p, epsilon, a=min_bound, b=max_bound))

class LinfMixin:
    distance = linf

    def get_epsilons(self, x: ep.Tensor, p: ep.Tensor, epsilon: float, min_bound: float, max_bound: float) -> ep.Tensor:
        norms = flatten(p).max(axis=-1)
        return epsilon / atleast_kd(norms, p.ndim)


class GaussianMixin:
    def sample_noise(self, x: ep.Tensor) -> ep.Tensor:
        return x.normal(x.shape)


class UniformMixin:
    def sample_noise(self, x: ep.Tensor) -> ep.Tensor:
        return x.uniform(x.shape, -1, 1)


class L2AdditiveGaussianNoiseAttack(L2Mixin, GaussianMixin, BaseAdditiveNoiseAttack):
    pass


class L2AdditiveUniformNoiseAttack(L2Mixin, UniformMixin, BaseAdditiveNoiseAttack):
    pass


class L2ClippingAwareAdditiveGaussianNoiseAttack(L2ClippingAwareMixin, GaussianMixin, BaseAdditiveNoiseAttack):
    pass


class L2ClippingAwareAdditiveUniformNoiseAttack(L2ClippingAwareMixin, UniformMixin, BaseAdditiveNoiseAttack):
    pass


class LinfAdditiveUniformNoiseAttack(LinfMixin, UniformMixin, BaseAdditiveNoiseAttack):
    pass


class BaseRepeatedAdditiveNoiseAttack(FixedEpsilonAttack, ABC):
    def __init__(self, *, repeats: int = 100, check_trivial: bool = True):
        self.repeats = repeats
        self.check_trivial = check_trivial

    def run(self, model: Model, inputs: T, criterion: Union[Criterion, Any] = None, *, epsilon: float, **kwargs: Any) -> T:
        raise_if_kwargs(kwargs)
        x0, restore_type = ep.astensor_(inputs)
        criterion_ = get_criterion(criterion)
        del inputs, criterion, kwargs
        verify_input_bounds(x0, model)
        is_adversarial = get_is_adversarial(criterion_, model)
        min_bound, max_bound = model.bounds
        result = x0
        if self.check_trivial:
            found = is_adversarial(result)
        else:
            found = ep.zeros(x0, len(result)).bool()
        for _ in range(self.repeats):
            if found.all():
                break
            p = self.sample_noise(x0)
            epsilons = self.get_epsilons(x0, p, epsilon, min_bound, max_bound)
            x = x0 + epsilons * p
            x = x.clip(min_bound, max_bound)
            is_adv = is_adversarial(x)
            is_new_adv = ep.logical_and(is_adv, ep.logical_not(found))
            result = ep.where(atleast_kd(is_new_adv, x.ndim), x, result)
            found = ep.logical_or(found, is_adv)
        return restore_type(result)

    @abstractmethod
    def sample_noise(self, x: ep.Tensor) -> ep.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_epsilons(self, x: ep.Tensor, p: ep.Tensor, epsilon: float, min_bound: float, max_bound: float) -> ep.Tensor:
        raise NotImplementedError


class L2RepeatedAdditiveGaussianNoiseAttack(L2Mixin, GaussianMixin, BaseRepeatedAdditiveNoiseAttack):
    pass


class L2RepeatedAdditiveUniformNoiseAttack(L2Mixin, UniformMixin, BaseRepeatedAdditiveNoiseAttack):
    pass


class L2ClippingAwareRepeatedAdditiveGaussianNoiseAttack(L2ClippingAwareMixin, GaussianMixin, BaseRepeatedAdditiveNoiseAttack):
    pass


class L2ClippingAwareRepeatedAdditiveUniformNoiseAttack(L2ClippingAwareMixin, UniformMixin, BaseRepeatedAdditiveNoiseAttack):
    pass


class LinfRepeatedAdditiveUniformNoiseAttack(LinfMixin, UniformMixin, BaseRepeatedAdditiveNoiseAttack):
    pass