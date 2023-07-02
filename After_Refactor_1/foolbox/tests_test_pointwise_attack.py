from typing import List, Any
import eagerpy as ep
import foolbox as fbn
import foolbox.attacks as fa
import pytest

from conftest import ModeAndDataAndDescription


def get_attack_id(attack: fa.Attack) -> str:
    return repr(attack)


def normalize_input(fmodel: fbn.Model, x: ep.Tensor) -> ep.Tensor:
    return (x - fmodel.bounds.lower) / (fmodel.bounds.upper - fmodel.bounds.lower)


def initialize_adversaries(fmodel: fbn.Model, attack: fa.Attack, x: ep.Tensor, y: ep.Tensor) -> ep.Tensor:
    init_attack = fa.SaltAndPepperNoiseAttack(steps=50)
    init_advs = init_attack.run(fmodel, x, y)
    return init_advs


def compute_norms(advs: ep.Tensor, x: ep.Tensor, p: int) -> ep.Tensor:
    return ep.norms.lp(flatten(advs - x), p=p, axis=-1)


def check_l2_norm_reduction(init_norms: ep.Tensor, norms: ep.Tensor) -> bool:
    return (norms < init_norms).any()


def run_untargeted_attack(fmodel: fbn.Model, x: ep.Tensor, y: ep.Tensor, attack: fa.PointwiseAttack) -> None:
    init_advs = initialize_adversaries(fmodel, attack, x, y)
    advs = attack.run(fmodel, x, y, starting_points=init_advs)

    init_norms_l0 = compute_norms(init_advs, x, p=0)
    norms_l0 = compute_norms(advs, x, p=0)
    init_norms_l2 = compute_norms(init_advs, x, p=2)
    norms_l2 = compute_norms(advs, x, p=2)

    is_smaller_l0 = check_l2_norm_reduction(init_norms_l0, norms_l0)
    is_smaller_l2 = check_l2_norm_reduction(init_norms_l2, norms_l2)

    assert fbn.accuracy(fmodel, advs, y) < fbn.accuracy(fmodel, x, y)
    assert fbn.accuracy(fmodel, advs, y) <= fbn.accuracy(fmodel, init_advs, y)
    assert is_smaller_l2
    assert is_smaller_l0


def run_targeted_attack(fmodel: fbn.Model, x: ep.Tensor, y: ep.Tensor, attack: fa.PointwiseAttack) -> None:
    init_advs = initialize_adversaries(fmodel, attack, x, y)

    logits = fmodel(init_advs)
    num_classes = logits.shape[-1]
    target_classes = logits.argmax(-1)
    target_classes = ep.where(
        target_classes == y, (target_classes + 1) % num_classes, target_classes
    )
    criterion = fbn.TargetedMisclassification(target_classes)

    advs = attack.run(fmodel, x, criterion, starting_points=init_advs)

    init_norms_l0 = compute_norms(init_advs, x, p=0)
    norms_l0 = compute_norms(advs, x, p=0)
    init_norms_l2 = compute_norms(init_advs, x, p=2)
    norms_l2 = compute_norms(advs, x, p=2)

    is_smaller_l0 = check_l2_norm_reduction(init_norms_l0, norms_l0)
    is_smaller_l2 = check_l2_norm_reduction(init_norms_l2, norms_l2)

    assert fbn.accuracy(fmodel, advs, y) < fbn.accuracy(fmodel, x, y)
    assert fbn.accuracy(fmodel, advs, y) <= fbn.accuracy(fmodel, init_advs, y)
    assert fbn.accuracy(fmodel, advs, target_classes) > fbn.accuracy(
        fmodel, x, target_classes
    )
    assert fbn.accuracy(fmodel, advs, target_classes) >= fbn.accuracy(
        fmodel, init_advs, target_classes
    )
    assert is_smaller_l2
    assert is_smaller_l0


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

    x = normalize_input(fmodel, x)
    fmodel = fmodel.transform_bounds((0, 1))

    run_untargeted_attack(fmodel, x, y, attack)


@pytest.mark.parametrize("attack", attacks, ids=get_attack_id)
def test_pointwise_targeted_attack(
    request: Any,
    fmodel_and_data_ext_for_attacks: ModeAndDataAndDescription,
    attack: fa.PointwiseAttack,
) -> None:
    (fmodel, x, y), real, low_dimensional_input = fmodel_and_data_ext_for_attacks

    if not low_dimensional_input or not real:
        pytest.skip()

    x = normalize_input(fmodel, x)
    fmodel = fmodel.transform_bounds((0, 1))

    run_targeted_attack(fmodel, x, y, attack)