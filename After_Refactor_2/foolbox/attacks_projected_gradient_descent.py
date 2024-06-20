from typing import Optional
import eagerpy as ep


class Optimizer:
    def __init__(self, x: ep.Tensor, stepsize: float):
        self.x = x
        self.stepsize = stepsize


class AdamOptimizer(Optimizer):
    def __init__(
        self,
        x: ep.Tensor,
        stepsize: float,
        adam_beta1: float,
        adam_beta2: float,
        adam_epsilon: float,
    ):
        super().__init__(x, stepsize)
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon


class BaseGradientDescent:
    def __init__(
        self,
        rel_stepsize: float = 0.025,
        abs_stepsize: Optional[float] = None,
        steps: int = 50,
        random_start: bool = True,
    ):
        self.rel_stepsize = rel_stepsize
        self.abs_stepsize = abs_stepsize
        self.steps = steps
        self.random_start = random_start

    def get_perturbation(self, model, inputs, labels):
        raise NotImplementedError("Subclass must implement get_perturbation method.")

    def get_optimizer(self, x: ep.Tensor, stepsize: float) -> Optimizer:
        raise NotImplementedError("Subclass must implement get_optimizer method.")


class L1ProjectedGradientDescentAttack(BaseGradientDescent):
    def get_perturbation(self, model, inputs, labels):
        # Implementation for L1 Norm
        ...


class L2ProjectedGradientDescentAttack(BaseGradientDescent):
    def get_perturbation(self, model, inputs, labels):
        # Implementation for L2 Norm
        ...


class LinfProjectedGradientDescentAttack(BaseGradientDescent):
    def get_perturbation(self, model, inputs, labels):
        # Implementation for Linf Norm
        ...


class L1AdamProjectedGradientDescentAttack(L1ProjectedGradientDescentAttack):
    def __init__(
        self,
        *,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-8,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon

    def get_optimizer(self, x: ep.Tensor, stepsize: float) -> Optimizer:
        return AdamOptimizer(
            x,
            stepsize,
            self.adam_beta1,
            self.adam_beta2,
            self.adam_epsilon,
        )


class L2AdamProjectedGradientDescentAttack(L2ProjectedGradientDescentAttack):
    def __init__(
        self,
        *,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-8,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon

    def get_optimizer(self, x: ep.Tensor, stepsize: float) -> Optimizer:
        return AdamOptimizer(
            x,
            stepsize,
            self.adam_beta1,
            self.adam_beta2,
            self.adam_epsilon,
        )


class LinfAdamProjectedGradientDescentAttack(LinfProjectedGradientDescentAttack):
    def __init__(
        self,
        *,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-8,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon

    def get_optimizer(self, x: ep.Tensor, stepsize: float) -> Optimizer:
        return AdamOptimizer(
            x,
            stepsize,
            self.adam_beta1,
            self.adam_beta2,
            self.adam_epsilon,
        )