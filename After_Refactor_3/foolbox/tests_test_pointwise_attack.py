from typing import List, Any

import foolbox as fbn
import foolbox.attacks as fa
import pytest

from conftest import ModeAndDataAndDescription
from foolbox.devutils import flatten


def get_attack_id(attack: fa.Attack) -> str:
    return repr(attack)


def initialize_attack(attack: fa.Attack, fmodel: fbn.models.Model, x: Any, y: Any) -> Any:
    init_attack = fa.SaltAndPepperNoiseAttack(steps=50)
    init_advs = init_attack.run(fmodel, x, y)
    advs = attack.run(fmodel, x, y, starting_points=init_advs)
    return init_advs, advs


def initialize_targeted_attack(
    attack: fa.Attack, fmodel: fbn.models.Model, x: Any, y: Any
) -> Any:
    init_attack = fa.SaltAndPepperNoiseAttack(steps=50)
    init_advs = init_attack.run(fmodel, x, y)
    logits = fmodel(init_advs)
    num_classes = logits.shape[-1]
    target_classes = logits.argmax(-1)
    target_classes = ep.where(
        target_classes == y, (target_classes + 1) % num_classes, target_classes
    )
    criterion = fbn.TargetedMisclassification(target_classes)
    advs = attack.run(fmodel, x, criterion, starting_points=init_advs)
    return init_advs, advs


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

    x_normalized = (x - fmodel.bounds.lower) / (fmodel.bounds.upper - fmodel.bounds.lower)
    fmodel = fmodel.transform_bounds((0, 1))

    init_advs, advs = initialize_attack(attack, fmodel, x_normalized, y)

    init_norms_l0 = ep.norms.lp(flatten(init_advs - x_normalized), p=0, axis=-1)
    norms_l0 = ep.norms.lp(flatten(advs - x_normalized), p=0, axis=-1)

    init_norms_l2 = ep.norms.lp(flatten(init_advs - x_normalized), p=2, axis=-1)
    norms_l2 = ep.norms.lp(flatten(advs - x_normalized), p=2, axis=-1)

    is_smaller_l0 = norms_l0 < init_norms_l0
    is_smaller_l2 = norms_l2 < init_norms_l2

    assert fbn.accuracy(fmodel, advs, y) < fbn.accuracy(fmodel, x_normalized, y)
    assert fbn.accuracy(fmodel, advs, y) <= fbn.accuracy(fmodel, init_advs, y)
    assert is_smaller_l2.any()
    assert is_smaller_l0.any()


@pytest.mark.parametrize("attack", attacks, ids=get_attack_id)
def test_pointwise_targeted_attack(
    request: Any,
    fmodel_and_data_ext_for_attacks: ModeAndDataAndDescription,
    attack: fa.PointwiseAttack,
) -> None:
    (fmodel, x, y), real, low_dimensional_input = fmodel_and_data_ext_for_attacks

    if not low_dimensional_input or not real:
        pytest.skip()

    x_normalized = (x - fmodel.bounds.lower) / (fmodel.bounds.upper - fmodel.bounds.lower)
    fmodel = fmodel.transform_bounds((0, 1))

    init_advs, advs = initialize_targeted_attack(attack, fmodel, x_normalized, y)

    init_norms_l0 = ep.norms.lp(flatten(init_advs - x_normalized), p=0, axis=-1)
    norms_l0 = ep.norms.lp(flatten(advs - x_normalized), p=0, axis=-1)

    init_norms_l2 = ep.norms.lp(flatten(init_advs - x_normalized), p=2, axis=-1)
    norms_l2 = ep.norms.lp(flatten(advs - x_normalized), p=2, axis=-1)

    is_smaller_l0 = norms_l0 < init_norms_l0
    is_smaller_l2 = norms_l2 < init_norms_l2

    assert fbn.accuracy(fmodel, advs, y) < fbn.accuracy(fmodel, x_normalized, y)
    assert fbn.accuracy(fmodel, advs, y) <= fbn.accuracy(fmodel, init_advs, y)
    assert fbn.accuracy(fmodel, advs, target_classes) > fbn.accuracy(
        fmodel, x_normalized, target_classes
    )
    assert fbn.accuracy(fmodel, advs, target_classes) >= fbn.accuracy(
        fmodel, init_advs, target_classes
    )
    assert is_smaller_l2.any()
    assert is_smaller_l0.any()