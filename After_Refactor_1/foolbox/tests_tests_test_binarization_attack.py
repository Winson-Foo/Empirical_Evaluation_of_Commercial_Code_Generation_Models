import pytest
from foolbox import accuracy
from foolbox.models import ThresholdingWrapper
from foolbox.devutils import flatten
from foolbox.attacks import BinarySearchContrastReductionAttack, BinarizationRefinementAttack
from conftest import ModeAndDataAndDescription


def find_adversarials(fmodel, x, y, starting_points, epsilons=None):
    advs, _, _ = attack(fmodel, x, y, starting_points, epsilons)
    assert accuracy(fmodel, advs, y) < accuracy(fmodel, x, y)
    return advs


def test_binarization_attack(fmodel_and_data_ext_for_attacks: ModeAndDataAndDescription) -> None:
    (fmodel, x, y), _, low_dimensional_input = fmodel_and_data_ext_for_attacks

    if not low_dimensional_input:
        pytest.skip()

    x = (x - fmodel.bounds.lower) / (fmodel.bounds.upper - fmodel.bounds.lower)
    fmodel = fmodel.transform_bounds((0, 1))
    fmodel = ThresholdingWrapper(fmodel, threshold=0.5)
    
    acc = accuracy(fmodel, x, y)
    assert acc > 0

    attack = BinarySearchContrastReductionAttack(target=0)
    advs = find_adversarials(fmodel, x, y, epsilons=None)

    attack2 = BinarizationRefinementAttack(threshold=0.5, included_in="upper")
    advs2 = find_adversarials(fmodel, x, y, starting_points=advs, epsilons=None)

    assert (fmodel(advs).argmax(axis=-1) == fmodel(advs2).argmax(axis=-1)).all()

    norms1 = flatten(advs - x).norms.l2(axis=-1)
    norms2 = flatten(advs2 - x).norms.l2(axis=-1)
    assert (norms2 <= norms1).all()
    assert (norms2 < norms1).any()

    attack2 = BinarizationRefinementAttack(included_in="upper")
    advs2 = find_adversarials(fmodel, x, y, starting_points=advs, epsilons=None)

    assert (fmodel(advs).argmax(axis=-1) == fmodel(advs2).argmax(axis=-1)).all()

    norms1 = flatten(advs - x).norms.l2(axis=-1)
    norms2 = flatten(advs2 - x).norms.l2(axis=-1)
    assert (norms2 <= norms1).all()
    assert (norms2 < norms1).any()

    with pytest.raises(ValueError, match="starting_points"):
        attack2(fmodel, x, y, epsilons=None)

    attack2 = BinarizationRefinementAttack(included_in="lower")
    with pytest.raises(ValueError, match="does not match"):
        attack2(fmodel, x, y, starting_points=advs, epsilons=None)

    attack2 = BinarizationRefinementAttack(included_in="invalid")  # type: ignore
    with pytest.raises(ValueError, match="expected included_in"):
        attack2(fmodel, x, y, starting_points=advs, epsilons=None)