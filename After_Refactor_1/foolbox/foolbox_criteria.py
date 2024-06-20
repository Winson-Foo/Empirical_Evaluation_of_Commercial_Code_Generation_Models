from typing import TypeVar, Any
from abc import ABC, abstractmethod
import eagerpy as ep

T = TypeVar("T")

class Criterion(ABC):
    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def __call__(self, perturbed: T, outputs: T) -> T:
        pass

    def __and__(self, other: "Criterion") -> "Criterion":
        return And(self, other)


class And(Criterion):
    def __init__(self, criterion_a: Criterion, criterion_b: Criterion):
        super().__init__()
        self.criterion_a = criterion_a
        self.criterion_b = criterion_b

    def __repr__(self) -> str:
        return f"{self.criterion_a!r} & {self.criterion_b!r}"

    def __call__(self, perturbed: T, outputs: T) -> T:
        perturbed, outputs = ep.astensors(perturbed, outputs)
        return self.criterion_a(perturbed, outputs) & self.criterion_b(perturbed, outputs)


class Misclassification(Criterion):
    def __init__(self, labels: Any):
        super().__init__()
        self.labels: ep.Tensor = ep.astensor(labels)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.labels!r})"

    def __call__(self, perturbed: T, outputs: T) -> T:
        outputs = ep.astensor(outputs)
        classes = outputs.argmax(axis=-1)
        return classes != self.labels


class TargetedMisclassification(Criterion):
    def __init__(self, target_classes: Any):
        super().__init__()
        self.target_classes: ep.Tensor = ep.astensor(target_classes)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.target_classes!r})"

    def __call__(self, perturbed: T, outputs: T) -> T:
        outputs = ep.astensor(outputs)
        classes = outputs.argmax(axis=-1)
        return classes == self.target_classes