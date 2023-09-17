from typing import Any
import eagerpy as ep


class Criterion:
    def __repr__(self) -> str:
        pass

    def __call__(self, perturbed, outputs):
        pass

    def __and__(self, other):
        return CriterionAnd(self, other)


class CriterionAnd(Criterion):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __repr__(self) -> str:
        return f"{self.a!r} & {self.b!r}"

    def __call__(self, perturbed, outputs):
        args, restore_type = ep.astensors_(perturbed, outputs)
        a = self.a(*args)
        b = self.b(*args)
        is_adv = ep.logical_and(a, b)
        return restore_type(is_adv)


class Misclassification(Criterion):
    def __init__(self, labels):
        self.labels = ep.astensor(labels)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.labels!r})"

    def __call__(self, perturbed, outputs):
        outputs_, restore_type = ep.astensor_(outputs)
        classes = outputs_.argmax(axis=-1)
        is_adv = classes != self.labels
        return restore_type(is_adv)


class TargetedMisclassification(Criterion):
    def __init__(self, target_classes):
        self.target_classes = ep.astensor(target_classes)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.target_classes!r})"

    def __call__(self, perturbed, outputs):
        outputs_, restore_type = ep.astensor_(outputs)
        classes = outputs_.argmax(axis=-1)
        is_adv = classes == self.target_classes
        return restore_type(is_adv)