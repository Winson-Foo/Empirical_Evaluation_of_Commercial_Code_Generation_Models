import pytest
from foolbox.models import ThresholdingWrapper
from foolbox import accuracy, devutils, attacks
from conftest import ModeAndDataAndDescription


def test_binarization_attack(fmodel_and_data_ext_for_attacks: ModeAndDataAndDescription) -> None:
    # get a model with thresholding
    (fmodel, x, y), _, low_dimensional_input = fmodel_and_data_ext_for_attacks

    # binarization doesn't work well for imagenet models
    if not low_dimensional_input:
        pytest.skip()

    x = (x - fmodel.bounds.lower) / (fmodel.bounds.upper - fmodel.bounds.lower)
    fmodel = fmodel.transform_bounds((0, 1))
    fmodel = ThresholdingWrapper(fmodel, threshold=0.5)
    acc = accuracy(fmodel, x, y)
    assert acc > 0

    # find some adversarials and check that they are non-trivial
    attack = attacks.BinarySearchContrastReductionAttack(target=0)
    advs = run_attack(attack, fmodel, x, y)
    assert accuracy(fmodel, advs, y) < acc

    # run the refinement attack with threshold=0.5 and inclusion as "upper"
    attack2 = attacks.BinarizationRefinementAttack(threshold=0.5, included_in="upper")
    advs2 = run_attack(attack2, fmodel, x, y, starting_points=advs)

    assert_classes_unchanged(fmodel, advs, advs2)
    assert_perturbations_decreased(advs, advs2)

    # run the refinement attack with inclusion as "upper"
    attack2 = attacks.BinarizationRefinementAttack(included_in="upper")
    advs2 = run_attack(attack2, fmodel, x, y, starting_points=advs)

    assert_classes_unchanged(fmodel, advs, advs2)
    assert_perturbations_decreased(advs, advs2)

    with pytest.raises(ValueError, match="starting_points"):
        attack2(fmodel, x, y, epsilons=None)

    attack2 = attacks.BinarizationRefinementAttack(included_in="lower")
    with pytest.raises(ValueError, match="does not match"):
        attack2(fmodel, x, y, starting_points=advs, epsilons=None)

    attack2 = attacks.BinarizationRefinementAttack(included_in="invalid")  # type: ignore
    with pytest.raises(ValueError, match="expected included_in"):
        attack2(fmodel, x, y, starting_points=advs, epsilons=None)


def run_attack(attack, fmodel, x, y, **kwargs):
    advs, _, _ = attack(fmodel, x, y, epsilons=None, **kwargs)
    return advs


def assert_classes_unchanged(fmodel, advs, advs2):
    assert (fmodel(advs).argmax(axis=-1) == fmodel(advs2).argmax(axis=-1)).all()


def assert_perturbations_decreased(advs, advs2):
    norms1 = devutils.flatten(advs - x).norms.l2(axis=-1)
    norms2 = devutils.flatten(advs2 - x).norms.l2(axis=-1)
    assert (norms2 <= norms1).all()
    assert (norms2 < norms1).any()