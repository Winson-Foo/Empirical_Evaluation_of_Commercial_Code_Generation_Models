import foolbox as fbn
import eagerpy as ep
from typing import Tuple


def test_correct_unperturbed(
    fmodel_and_data: Tuple[fbn.Model, ep.Tensor, ep.Tensor]
) -> None:
    fmodel, inputs, _ = fmodel_and_data
    perturbed = inputs
    logits = fmodel(perturbed)
    labels = logits.argmax(axis=-1)

    assert not is_any_misclassified(perturbed, logits, labels)
    assert not is_any_targeted_misclassified(perturbed, logits, labels)
    assert not is_any_combined_misclassification(perturbed, logits, labels)


def test_wrong_unperturbed(
    fmodel_and_data: Tuple[fbn.Model, ep.Tensor, ep.Tensor]
) -> None:
    fmodel, inputs, _ = fmodel_and_data
    perturbed = inputs
    logits = fmodel(perturbed)
    num_classes = logits.shape[1]
    labels = (logits.argmax(axis=-1) + 1) % num_classes

    assert is_all_misclassified(perturbed, logits, labels)
    assert is_targeted_misclassified(num_classes, perturbed, logits, labels)
    assert is_all_combined_misclassification(perturbed, logits, labels)
    assert not is_any_combined_targeted_misclassification(num_classes, perturbed, logits, labels)


def is_any_misclassified(
    perturbed: ep.Tensor, logits: ep.Tensor, labels: ep.Tensor
) -> bool:
    return fbn.Misclassification(labels)(perturbed, logits).any()


def is_any_targeted_misclassified(
    perturbed: ep.Tensor, logits: ep.Tensor, labels: ep.Tensor
) -> bool:
    num_classes = logits.shape[1]
    target_classes = (labels + 1) % num_classes
    return fbn.TargetedMisclassification(target_classes)(perturbed, logits).any()


def is_any_combined_misclassification(
    perturbed: ep.Tensor, logits: ep.Tensor, labels: ep.Tensor
) -> bool:
    combined = fbn.Misclassification(labels) & fbn.Misclassification(labels)
    return combined(perturbed, logits).any()


def is_all_misclassified(
    perturbed: ep.Tensor, logits: ep.Tensor, labels: ep.Tensor
) -> bool:
    return fbn.Misclassification(labels)(perturbed, logits).all()


def is_targeted_misclassified(
    num_classes: int, perturbed: ep.Tensor, logits: ep.Tensor, labels: ep.Tensor
) -> bool:
    target_classes = (labels + 1) % num_classes
    targeted_misclassification = fbn.TargetedMisclassification(target_classes)
    if num_classes > 2:
        return not targeted_misclassification(perturbed, logits).any()
    else:
        return targeted_misclassification(perturbed, logits).all()


def is_all_combined_misclassification(
    perturbed: ep.Tensor, logits: ep.Tensor, labels: ep.Tensor
) -> bool:
    combined = fbn.Misclassification(labels) & fbn.Misclassification(labels)
    return combined(perturbed, logits).all()


def is_any_combined_targeted_misclassification(
    num_classes: int, perturbed: ep.Tensor, logits: ep.Tensor, labels: ep.Tensor
) -> bool:
    target_classes = (labels + 1) % num_classes
    combined = fbn.TargetedMisclassification(labels) & fbn.TargetedMisclassification(target_classes)
    return combined(perturbed, logits).any()


def test_repr_object() -> None:
    assert repr(object()).startswith("<")


def test_repr_misclassification(dummy: ep.Tensor) -> None:
    labels = ep.arange(dummy, 10)
    assert not repr(fbn.Misclassification(labels)).startswith("<")


def test_repr_and(dummy: ep.Tensor) -> None:
    labels = ep.arange(dummy, 10)
    assert not repr(
        fbn.Misclassification(labels) & fbn.Misclassification(labels)
    ).startswith("<")


def test_repr_targeted_misclassification(dummy: ep.Tensor) -> None:
    target_classes = ep.arange(dummy, 10)
    assert not repr(fbn.TargetedMisclassification(target_classes)).startswith("<")