from abc import ABC, abstractmethod

from .devutils import flatten
from .devutils import atleast_kd


class Distance(ABC):
    @abstractmethod
    def __call__(self, references, perturbed):
        """
        Calculates the distances from references to perturbed.

        Args:
            references: A batch of reference inputs.
            perturbed: A batch of perturbed inputs.

        Returns:
            A 1D tensor with the distances from references to perturbed.
        """
        ...

    @abstractmethod
    def clip_perturbation(self, references, perturbed, epsilon):
        """
        Clips the perturbations to epsilon and returns the new perturbed.

        Args:
            references: A batch of reference inputs.
            perturbed: A batch of perturbed inputs.
            epsilon: The maximum allowed perturbation.

        Returns:
            A tensor like perturbed but with the perturbation clipped to epsilon.
        """
        ...


class LpDistance(Distance):
    def __init__(self, p):
        self.p = p

    def __repr__(self):
        return f"LpDistance({self.p})"

    def __str__(self):
        return f"L{self.p} distance"

    def __call__(self, references, perturbed):
        """Calculates the distances from references to perturbed using the Lp norm.

        Args:
            references: A batch of reference inputs.
            perturbed: A batch of perturbed inputs.

        Returns:
            A 1D tensor with the distances from references to perturbed.
        """
        (x, y), restore_type = ep.astensors_(references, perturbed)
        norms = ep.norms.lp(flatten(y - x), self.p, axis=-1)
        return restore_type(norms)

    def clip_perturbation(self, references, perturbed, epsilon):
        """Clips the perturbations to epsilon and returns the new perturbed.

        Args:
            references: A batch of reference inputs.
            perturbed: A batch of perturbed inputs.
            epsilon: The maximum allowed perturbation.

        Returns:
            A tensor like perturbed but with the perturbation clipped to epsilon.
        """
        (x, y), restore_type = ep.astensors_(references, perturbed)
        p = y - x
        if self.p == ep.inf:
            clipped_perturbation = ep.clip(p, -epsilon, epsilon)
        else:
            norms = ep.norms.lp(flatten(p), self.p, axis=-1)
            norms = ep.maximum(norms, 1e-12)  # avoid division by zero
            factor = epsilon / norms
            factor = ep.minimum(1, factor)  # clipping -> decreasing but not increasing
            if self.p == 0:
                if (factor == 1).all():
                    return perturbed
                raise NotImplementedError("reducing L0 norms not yet supported")
            factor = atleast_kd(factor, x.ndim)
            clipped_perturbation = factor * p
        return restore_type(x + clipped_perturbation)


l0 = LpDistance(0)
l1 = LpDistance(1)
l2 = LpDistance(2)
linf = LpDistance(ep.inf)