from typing import Union, Any
import eagerpy as ep
from abc import ABC, abstractmethod

from foolbox_devutils import flatten, atleast_kd
from foolbox.distances import l2, linf
from foolbox.attacks.base import FixedEpsilonAttack, Criterion, Model, T
from foolbox.utils import raise_if_kwargs
from foolbox.attacks.external.clipping_aware_rescaling import (
    l2_clipping_aware_rescaling,
)
from foolbox.attacks.helpers import verify_input_bounds


class BaseAdditiveNoiseAttack(FixedEpsilonAttack, ABC):
    @abstractmethod
    def sample_noise(self, x: ep.Tensor) -> ep.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_epsilons(
        self, x: ep.Tensor, p: ep.Tensor, epsilon: float, min_: float, max_: float
    ) -> ep.Tensor:
        raise NotImplementedError

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Criterion, Any] = None,
        *,
        epsilon: float,
        **kwargs: Any,
    ) -> T:
        raise_if_kwargs(kwargs)
        x, restore_type = ep.astensor_(inputs)

        verify_input_bounds(x, model)
        min_, max_ = model.bounds

        p = self.sample_noise(x)
        epsilons = self.get_epsilons(x, p, epsilon, min_=min_, max_=max_)
        x = x + epsilons * p
        x = x.clip(min_, max_)

        return restore_type(x)


class L2Mixin:
    distance = l2

    def get_epsilons(
        self, x: ep.Tensor, p: ep.Tensor, epsilon: float, min_: float, max_: float
    ) -> ep.Tensor:
        norms = flatten(p).norms.l2(axis=-1)
        return epsilon / atleast_kd(norms, p.ndim)


class L2ClippingAwareMixin:
    distance = l2

    def get_epsilons(
        self, x: ep.Tensor, p: ep.Tensor, epsilon: float, min_: float, max_: float
    ) -> ep.Tensor:
        return l2_clipping_aware_rescaling(x, p, epsilon, a=min_, b=max_)


class LinfMixin:
    distance = linf

    def get_epsilons(
        self, x: ep.Tensor, p: ep.Tensor, epsilon: float, min_: float, max_: float
    ) -> ep.Tensor:
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


class L2ClippingAwareAdditiveGaussianNoiseAttack(
    L2ClippingAwareMixin, GaussianMixin, BaseAdditiveNoiseAttack
):
    pass


class L2ClippingAwareAdditiveUniformNoiseAttack(
    L2ClippingAwareMixin, UniformMixin, BaseAdditiveNoiseAttack
):
    pass


class LinfAdditiveUniformNoiseAttack(LinfMixin, UniformMixin, BaseAdditiveNoiseAttack):
    pass


class BaseRepeatedAdditiveNoiseAttack(FixedEpsilonAttack, ABC):
    def __init__(self, *, repeats: int = 100, check_trivial: bool = True):
        self.repeats = repeats
        self.check_trivial = check_trivial

    @abstractmethod
    def sample_noise(self, x: ep.Tensor) -> ep.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_epsilons(
        self, x: ep.Tensor, p: ep.Tensor, epsilon: float, min_: float, max_: float
    ) -> ep.Tensor:
        raise NotImplementedError

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Criterion, Any] = None,
        *,
        epsilon: float,
        **kwargs: Any,
    ) -> T:
        raise_if_kwargs(kwargs)
        x0, restore_type = ep.astensor_(inputs)

        criterion_ = get_criterion(criterion)
        verify_input_bounds(x0, model)
        min_, max_ = model.bounds

        result = x0
        if self.check_trivial:
            found = is_adversarial(result)
        else:
            found = ep.zeros(x0, len(result)).bool()

        for _ in range(self.repeats):
            if found.all():
                break

            p = self.sample_noise(x0)
            epsilons = self.get_epsilons(x0, p, epsilon, min_=min_, max_=max_)
            x = x0 + epsilons * p
            x = x.clip(min_, max_)
            is_adv = is_adversarial(x)
            is_new_adv = ep.logical_and(is_adv, ep.logical_not(found))
            result = ep.where(atleast_kd(is_new_adv, x.ndim), x, result)
            found = ep.logical_or(found, is_adv)

        return restore_type(result)


class L2RepeatedAdditiveGaussianNoiseAttack(
    L2Mixin, GaussianMixin, BaseRepeatedAdditiveNoiseAttack
):
    pass


class L2RepeatedAdditiveUniformNoiseAttack(
    L2Mixin, UniformMixin, BaseRepeatedAdditiveNoiseAttack
):
    pass


class L2ClippingAwareRepeatedAdditiveGaussianNoiseAttack(
    L2ClippingAwareMixin, GaussianMixin, BaseRepeatedAdditiveNoiseAttack
):
    pass


class L2ClippingAwareRepeatedAdditiveUniformNoiseAttack(
    L2ClippingAwareMixin, UniformMixin, BaseRepeatedAdditiveNoiseAttack
):
    pass


class LinfRepeatedAdditiveUniformNoiseAttack(
    LinfMixin, UniformMixin, BaseRepeatedAdditiveNoiseAttack
):
    pass