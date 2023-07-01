from functools import partial
import numpy as np
import eagerpy as ep

from ..models import Model
from ..criteria import Misclassification, TargetedMisclassification
from ..distances import l2
from ..types import Bounds
from .base import MinimizationAttack, T, get_criterion, raise_if_kwargs, verify_input_bounds
from .gradient_descent_base import AdamOptimizer


class L2CarliniWagnerAttack(MinimizationAttack):
    distance = l2

    def __init__(
        self,
        binary_search_steps=9,
        steps=10000,
        stepsize=1e-2,
        confidence=0,
        initial_const=1e-3,
        abort_early=True,
    ):
        self.binary_search_steps = binary_search_steps
        self.steps = steps
        self.stepsize = stepsize
        self.confidence = confidence
        self.initial_const = initial_const
        self.abort_early = abort_early

    def run(self, model: Model, inputs: T, criterion, *, early_stop=None, **kwargs):
        raise_if_kwargs(kwargs)
        x, restore_type = ep.astensor_(inputs)
        criterion_ = get_criterion(criterion)
        del inputs, criterion, kwargs

        verify_input_bounds(x, model)

        N = len(x)

        if isinstance(criterion_, Misclassification):
            targeted = False
            classes = criterion_.labels
            change_classes_logits = self.confidence
        elif isinstance(criterion_, TargetedMisclassification):
            targeted = True
            classes = criterion_.target_classes
            change_classes_logits = -self.confidence
        else:
            raise ValueError("unsupported criterion")

        def is_adversarial(perturbed, logits):
            if change_classes_logits != 0:
                logits += ep.onehot_like(logits, classes, value=change_classes_logits)
            return criterion_(perturbed, logits)

        if classes.shape != (N,):
            name = "target_classes" if targeted else "labels"
            raise ValueError(
                f"expected {name} to have shape ({N},), got {classes.shape}"
            )

        bounds = model.bounds
        to_attack_space = partial(self._to_attack_space, bounds=bounds)
        to_model_space = partial(self._to_model_space, bounds=bounds)

        x_attack = to_attack_space(x)
        reconstructed_x = to_model_space(x_attack)

        rows = range(N)

        def loss_fun(delta, consts):
            assert delta.shape == x_attack.shape
            assert consts.shape == (N,)

            x = to_model_space(x_attack + delta)
            logits = model(x)

            if targeted:
                c_minimize = self.best_other_classes(logits, classes)
                c_maximize = classes
            else:
                c_minimize = classes
                c_maximize = self.best_other_classes(logits, classes)

            is_adv_loss = logits[rows, c_minimize] - logits[rows, c_maximize]
            assert is_adv_loss.shape == (N,)

            is_adv_loss = is_adv_loss + self.confidence
            is_adv_loss = ep.maximum(0, is_adv_loss)
            is_adv_loss = is_adv_loss * consts

            squared_norms = self.flatten(x - reconstructed_x).square().sum(axis=-1)
            loss = is_adv_loss.sum() + squared_norms.sum()
            return loss, (x, logits)

        loss_aux_and_grad = ep.value_and_grad_fn(x, loss_fun, has_aux=True)

        consts = self.initial_const * np.ones((N,))
        lower_bounds = np.zeros((N,))
        upper_bounds = np.inf * np.ones((N,))

        best_advs = ep.zeros_like(x)
        best_advs_norms = ep.full(x, (N,), ep.inf)

        for binary_search_step in range(self.binary_search_steps):
            if (
                binary_search_step == self.binary_search_steps - 1
                and self.binary_search_steps >= 10
            ):
                consts = np.minimum(upper_bounds, 1e10)

            delta = ep.zeros_like(x_attack)
            optimizer = AdamOptimizer(delta, self.stepsize)

            found_advs = np.full((N,), fill_value=False)
            loss_at_previous_check = np.inf

            consts_ = ep.from_numpy(x, consts.astype(np.float32))

            for step in range(self.steps):
                loss, (perturbed, logits), gradient = loss_aux_and_grad(delta, consts_)
                delta -= optimizer(gradient)

                if self.abort_early and step % (np.ceil(self.steps / 10)) == 0:
                    if not (loss <= 0.9999 * loss_at_previous_check):
                        break
                    loss_at_previous_check = loss

                found_advs_iter = is_adversarial(perturbed, logits)
                found_advs = np.logical_or(found_advs, found_advs_iter.numpy())

                norms = self.flatten(perturbed - x).norms.l2(axis=-1)
                closer = norms < best_advs_norms
                new_best = ep.logical_and(closer, found_advs_iter)

                new_best_ = self.atleast_kd(new_best, best_advs.ndim)
                best_advs = ep.where(new_best_, perturbed, best_advs)
                best_advs_norms = ep.where(new_best, norms, best_advs_norms)

            upper_bounds = np.where(found_advs, consts, upper_bounds)
            lower_bounds = np.where(found_advs, lower_bounds, consts)

            consts_exponential_search = consts * 10
            consts_binary_search = (lower_bounds + upper_bounds) / 2
            consts = np.where(
                np.isinf(upper_bounds), consts_exponential_search, consts_binary_search
            )

        return restore_type(best_advs)

    @staticmethod
    def best_other_classes(logits, exclude):
        other_logits = logits - ep.onehot_like(logits, exclude, value=ep.inf)
        return other_logits.argmax(axis=-1)

    @staticmethod
    def _to_attack_space(x, *, bounds):
        min_, max_ = bounds
        a = (min_ + max_) / 2
        b = (max_ - min_) / 2
        x = (x - a) / b
        x = x * 0.999999
        x = x.arctanh()
        return x

    @staticmethod
    def _to_model_space(x, *, bounds):
        min_, max_ = bounds
        x = x.tanh()
        a = (min_ + max_) / 2
        b = (max_ - min_) / 2
        x = x * b + a
        return x

    @staticmethod
    def flatten(tensor):
        return ep.reshape(tensor, (tensor.shape[0], -1))

    @staticmethod
    def atleast_kd(tensor, k):
        while len(tensor.shape) < k:
            tensor = ep.expand_dims(tensor, -1)
        return tensor


def _main():
    # Example usage of L2CarliniWagnerAttack
    pass


if __name__ == "__main__":
    _main()