from typing import List, Any
import eagerpy as ep
import foolbox as fbn
import foolbox.attacks as fa
import pytest

from conftest import ModeAndDataAndDescription


def get_attack_id(x: fa.Attack) -> str:
    return repr(x)


def preprocess_input(fmodel: fbn.Model, x: ep.Tensor) -> ep.Tensor:
    x = (x - fmodel.bounds.lower) / (fmodel.bounds.upper - fmodel.bounds.lower)
    return fmodel.transform_bounds((0, 1))


def run_init_attack(fmodel: fbn.Model, x: ep.Tensor, y: ep.Tensor) -> ep.Tensor:
    init_attack = fa.SaltAndPepperNoiseAttack(steps=50)
    init_advs = init_attack.run(fmodel, x, y)
    return init_advs


def run_attack(fmodel: fbn.Model, x: ep.Tensor, criterion: fbn.criteria.Criterion, starting_points: ep.Tensor) -> ep.Tensor:
    return attack.run(fmodel, x, criterion, starting_points=starting_points)


def assert_accuracy_below_threshold(fmodel: fbn.Model, advs: ep.Tensor, x: ep.Tensor, y: ep.Tensor, threshold: float) -> None:
    assert fbn.accuracy(fmodel, advs, y) <= threshold


def assert_accuracy_above_threshold(fmodel: fbn.Model, advs: ep.Tensor, x: ep.Tensor, y: ep.Tensor, threshold: float) -> None:
    assert fbn.accuracy(fmodel, advs, y) > threshold


def assert_l2_norm_smaller(advs: ep.Tensor, x: ep.Tensor, init_advs: ep.Tensor) -> None:
    init_norms_l2 = ep.norms.lp(flatten(init_advs - x), p=2, axis=-1)
    norms_l2 = ep.norms.lp(flatten(advs - x), p=2, axis=-1)
    assert (norms_l2 < init_norms_l2).any()


def assert_l0_norm_smaller(advs: ep.Tensor, x: ep.Tensor, init_advs: ep.Tensor) -> None:
    init_norms_l0 = ep.norms.lp(flatten(init_advs - x), p=0, axis=-1)
    norms_l0 = ep.norms.lp(flatten(advs - x), p=0, axis=-1)
    assert (norms_l0 < init_norms_l0).any()


attacks: List[fa.Attack] = [
    fa.PointwiseAttack(),
    fa.PointwiseAttack(l2_binary_search=False),
]


@pytest.mark.parametrize("attack", attacks, ids=get_attack_id)
def test_pointwise_untargeted_attack(
    request: Any,
    fmodel_and_data_ext_for_attacks: ModeAndDataAndDescription,
    attack: fa.PointwiseAttack,
) -> None:
    (fmodel, x, y), real, low_dimensional_input = fmodel_and_data_ext_for_attacks

    if not low_dimensional_input or not real:
        pytest.skip()

    x = preprocess_input(fmodel, x)
    init_advs = run_init_attack(fmodel, x, y)
    advs = run_attack(fmodel, x, y, init_advs)

    assert_accuracy_below_threshold(fmodel, advs, x, y, fbn.accuracy(fmodel, x, y))
    assert_l2_norm_smaller(advs, x, init_advs)
    assert_l0_norm_smaller(advs, x, init_advs)


@pytest.mark.parametrize("attack", attacks, ids=get_attack_id)
def test_pointwise_targeted_attack(
    request: Any,
    fmodel_and_data_ext_for_attacks: ModeAndDataAndDescription,
    attack: fa.PointwiseAttack,
) -> None:
    (fmodel, x, y), real, low_dimensional_input = fmodel_and_data_ext_for_attacks

    if not low_dimensional_input or not real:
        pytest.skip()

    x = preprocess_input(fmodel, x)
    init_advs = run_init_attack(fmodel, x, y)

    logits = fmodel(init_advs)
    num_classes = logits.shape[-1]
    target_classes = logits.argmax(-1)
    target_classes = ep.where(
        target_classes == y, (target_classes + 1) % num_classes, target_classes
    )
    criterion = fbn.TargetedMisclassification(target_classes)

    advs = run_attack(fmodel, x, criterion, init_advs)

    assert_accuracy_below_threshold(fmodel, advs, x, y, fbn.accuracy(fmodel, x, y))
    assert_accuracy_above_threshold(fmodel, advs, x, target_classes, fbn.accuracy(fmodel, x, target_classes))
    assert_l2_norm_smaller(advs, x, init_advs)
    assert_l0_norm_smaller(advs, x, init_advs)