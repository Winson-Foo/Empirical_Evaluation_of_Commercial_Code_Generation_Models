from typing import TypeVar, Any
from abc import ABC, abstractmethod
import eagerpy as ep

T = TypeVar("T")

class Criterion(ABC):
    @abstractmethod
    def __call__(self, perturbed: T, outputs: T) -> T:
        ...

    def __and__(self, other: "Criterion") -> "Criterion":
        return _And(self, other)

class _And(Criterion):
    def __init__(self, criterion1: Criterion, criterion2: Criterion):
        super().__init__()
        self.criterion1 = criterion1
        self.criterion2 = criterion2

    def __repr__(self) -> str:
        return f"{self.criterion1!r} & {self.criterion2!r}"

    def __call__(self, perturbed: T, outputs: T) -> T:
        args, restore_type = ep.astensors_(perturbed, outputs)
        a = self.criterion1(*args)
        b = self.criterion2(*args)
        is_adv = ep.logical_and(a, b)
        return restore_type(is_adv)

class Misclassification(Criterion):
    def __init__(self, labels: Any):
        super().__init__()
        self.labels = ep.astensor(labels)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.labels!r})"

    def __call__(self, perturbed: T, outputs: T) -> T:
        outputs_, restore_type = ep.astensor_(outputs)
        del perturbed

        classes = outputs_.argmax(axis=-1)
        assert classes.shape == self.labels.shape
        is_adv = classes != self.labels
        return restore_type(is_adv)

class TargetedMisclassification(Criterion):
    def __init__(self, target_classes: Any):
        super().__init__()
        self.target_classes = ep.astensor(target_classes)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.target_classes!r})"

    def __call__(self, perturbed: T, outputs: T) -> T:
        outputs_, restore_type = ep.astensor_(outputs)
        del perturbed

        classes = outputs_.argmax(axis=-1)
        assert classes.shape == self.target_classes.shape
        is_adv = classes == self.target_classes
        return restore_type(is_adv)