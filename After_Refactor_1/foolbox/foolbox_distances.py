from abc import ABC, abstractmethod
from typing import TypeVar
import eagerpy as ep

from .devutils import flatten, atleast_kd


T = TypeVar("T")


class Distance(ABC):
    @abstractmethod
    def __call__(self, reference: T, perturbed: T) -> T:
        pass

    @abstractmethod
    def clip_perturbation(self, references: T, perturbed: T, epsilon: float) -> T:
        pass


class LpDistance(Distance):
    def __init__(self, p: float):
        self.p = p

    def __repr__(self) -> str:
        return f"LpDistance({self.p})"

    def __str__(self) -> str:
        return f"L{self.p} distance"

    def calculate_distances(self, reference: T, perturbed: T) -> T:
        (x, y), restore_type = ep.astensors_(reference, perturbed)
        norms = ep.norms.lp(flatten(y - x), self.p, axis=-1)
        return restore_type(norms)

    def __call__(self, reference: T, perturbed: T) -> T:
        """Calculates the distances from references to perturbed using the Lp norm.

        Args:
            reference: A batch of reference inputs.
            perturbed: A batch of perturbed inputs.

        Returns:
            A 1D tensor with the distances from references to perturbed.
        """
        return self.calculate_distances(reference, perturbed)

    def clip_perturbation(self, reference: T, perturbed: T, epsilon: float) -> T:
        """Clips the perturbations to epsilon and returns the new perturbed

        Args:
            reference: A batch of reference inputs.
            perturbed: A batch of perturbed inputs.

        Returns:
            A tensor like perturbed but with the perturbation clipped to epsilon.
        """
        (x, y), restore_type = ep.astensors_(reference, perturbed)
        p = y - x
        if self.p == ep.inf:
            clipped_perturbation = ep.clip(p, -epsilon, epsilon)
            return restore_type(x + clipped_perturbation)
        norms = ep.norms.lp(flatten(p), self.p, axis=-1)
        norms = ep.maximum(norms, 1e-12)  # avoid divsion by zero
        factor = epsilon / norms
        factor = ep.minimum(1, factor)  # clipping -> decreasing but not increasing
        if self.p == 0:
            if (factor == 1).all():
                return perturbed
            raise NotImplementedError("Reducing L0 norms not yet supported")
        factor = atleast_kd(factor, x.ndim)
        clipped_perturbation = factor * p
        return restore_type(x + clipped_perturbation)


l0_distance = LpDistance(0)
l1_distance = LpDistance(1)
l2_distance = LpDistance(2)
linf_distance = LpDistance(ep.inf)