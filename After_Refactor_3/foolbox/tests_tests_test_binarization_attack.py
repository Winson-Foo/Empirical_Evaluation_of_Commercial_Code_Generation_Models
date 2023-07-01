import pytest
from foolbox import accuracy
from foolbox.models import ThresholdingWrapper
from foolbox.attacks import (
    BinarySearchContrastReductionAttack,
    BinarizationRefinementAttack,
)
from foolbox.devutils import flatten
from conftest import ModeAndDataAndDescription


def test_binarization_attack(fmodel_and_data_ext_for_attacks: ModeAndDataAndDescription) -> None:
    fmodel, x, y = fmodel_and_data_ext_for_attacks

    # Skip if low-dimensional input is False
    if not fmodel_and_data_ext_for_attacks.low_dimensional_input:
        pytest.skip()

    x = (x - fmodel.bounds.lower) / (fmodel.bounds.upper - fmodel.bounds.lower)
    fmodel = fmodel.transform_bounds((0, 1))
    fmodel = ThresholdingWrapper(fmodel, threshold=0.5)
    initial_acc = accuracy(fmodel, x, y)
    assert initial_acc > 0

    # Find adversarials and check non-triviality
    attack = BinarySearchContrastReductionAttack(target=0)
    adversarials, _, _ = attack(fmodel, x, y, epsilons=None)
    assert accuracy(fmodel, adversarials, y) < initial_acc

    # Run the refinement attack
    refinement_attack = BinarizationRefinementAttack(threshold=0.5, included_in="upper")
    adversarials_refined, _, _ = refinement_attack(fmodel, x, y, starting_points=adversarials, epsilons=None)

    # Ensure predicted classes didn't change
    assert (fmodel(adversarials).argmax(axis=-1) == fmodel(adversarials_refined).argmax(axis=-1)).all()

    # Ensure perturbations didn't get larger and some got smaller
    norms1 = flatten(adversarials - x).norms.l2(axis=-1)
    norms2 = flatten(adversarials_refined - x).norms.l2(axis=-1)
    assert (norms2 <= norms1).all()
    assert (norms2 < norms1).any()

    # Run the refinement attack without specifying threshold
    refinement_attack = BinarizationRefinementAttack(included_in="upper")
    adversarials_refined, _, _ = refinement_attack(fmodel, x, y, starting_points=adversarials, epsilons=None)

    # Ensure predicted classes didn't change
    assert (fmodel(adversarials).argmax(axis=-1) == fmodel(adversarials_refined).argmax(axis=-1)).all()

    # Ensure perturbations didn't get larger and some got smaller
    norms1 = flatten(adversarials - x).norms.l2(axis=-1)
    norms2 = flatten(adversarials_refined - x).norms.l2(axis=-1)
    assert (norms2 <= norms1).all()
    assert (norms2 < norms1).any()

    # Check for ValueError when 'starting_points' not provided
    with pytest.raises(ValueError, match="starting_points"):
        refinement_attack(fmodel, x, y, epsilons=None)

    # Check for ValueError when 'included_in' doesn't match
    refinement_attack = BinarizationRefinementAttack(included_in="lower")
    with pytest.raises(ValueError, match="does not match"):
        refinement_attack(fmodel, x, y, starting_points=adversarials, epsilons=None)

    # Check for ValueError when 'included_in' is invalid
    refinement_attack = BinarizationRefinementAttack(included_in="invalid")
    with pytest.raises(ValueError, match="expected included_in"):
        refinement_attack(fmodel, x, y, starting_points=adversarials, epsilons=None)