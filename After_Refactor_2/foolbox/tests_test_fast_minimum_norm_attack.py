from typing import Tuple, Union, List
import eagerpy as ep
import foolbox as fbn
import foolbox.attacks as fa
from foolbox.devutils import flatten
from foolbox.attacks.fast_minimum_norm import FMNAttackLp
import pytest
import numpy as np
from conftest import ModeAndDataAndDescription


def get_attack_id(x: Tuple[FMNAttackLp, Union[int, float]]) -> str:
    return repr(x[0])


ATTACKS: List[Tuple[fa.Attack, Union[int, float]]] = [
    (fa.L0FMNAttack(steps=20), 0),
    (fa.L1FMNAttack(steps=20), 1),
    (fa.L2FMNAttack(steps=20), 2),
    (fa.LInfFMNAttack(steps=20), ep.inf),
    (fa.LInfFMNAttack(steps=20, min_stepsize=1.0 / 100), ep.inf),
]


@pytest.mark.parametrize("attack_and_p", ATTACKS, ids=get_attack_id)
def test_fast_minimum_norm_untargeted_attack(
    fmodel_and_data_ext_for_attacks: ModeAndDataAndDescription,
    attack_and_p: Tuple[FMNAttackLp, Union[int, float]],
) -> None:

    (fmodel, x, y), real, low_dimensional_input = fmodel_and_data_ext_for_attacks

    if isinstance(x, ep.NumPyTensor):
        pytest.skip()

    x = normalize_input(x, fmodel)
    fmodel = fmodel.transform_bounds((0, 1))

    init_advs = get_initial_adversaries(fmodel, x)

    attack, p = attack_and_p
    advs = attack.run(fmodel, x, y, starting_points=init_advs)

    init_norms = get_norms(flatten(init_advs - x), p)
    norms = get_norms(flatten(advs - x), p)

    is_smaller = norms < init_norms

    assert fbn.accuracy(fmodel, advs, y) < fbn.accuracy(fmodel, x, y)
    assert fbn.accuracy(fmodel, advs, y) <= fbn.accuracy(fmodel, init_advs, y)
    assert is_smaller.any()


@pytest.mark.parametrize("attack_and_p", ATTACKS, ids=get_attack_id)
def test_fast_minimum_norm_targeted_attack(
    fmodel_and_data_ext_for_attacks: ModeAndDataAndDescription,
    attack_and_p: Tuple[FMNAttackLp, Union[int, float]],
) -> None:

    (fmodel, x, y), real, low_dimensional_input = fmodel_and_data_ext_for_attacks

    if isinstance(x, ep.NumPyTensor):
        pytest.skip()

    x = normalize_input(x, fmodel)
    fmodel = fmodel.transform_bounds((0, 1))

    target_classes = get_target_classes(y, fmodel)
    criterion = fbn.TargetedMisclassification(target_classes)
    adv_before_attack = criterion(x, fmodel(x))
    assert not adv_before_attack.all()

    init_advs = get_initial_adversaries(fmodel, x)

    attack, p = attack_and_p
    advs = attack.run(fmodel, x, criterion, starting_points=init_advs)

    init_norms = get_norms(flatten(init_advs - x), p)
    norms = get_norms(flatten(advs - x), p)

    is_smaller = norms < init_norms

    assert fbn.accuracy(fmodel, advs, target_classes) == 1.0
    assert is_smaller.any()


def normalize_input(x: ep.Tensor, fmodel: fbn.models.Model) -> ep.Tensor:
    return (x - fmodel.bounds.lower) / (fmodel.bounds.upper - fmodel.bounds.lower)


def get_initial_adversaries(fmodel: fbn.models.Model, x: ep.Tensor) -> ep.Tensor:
    init_attack = fa.DatasetAttack()
    init_attack.feed(fmodel, x)
    return init_attack.run(fmodel, x)


def get_norms(x: ep.Tensor, p: Union[int, float]) -> ep.Tensor:
    return ep.norms.lp(x, p=p, axis=-1)

def get_target_classes(y: ep.Tensor, fmodel: fbn.models.Model) -> ep.Tensor:
    unique_preds = np.unique(fmodel(y).argmax(-1).numpy())
    return ep.from_numpy(
        y,
        np.array(
            [
                unique_preds[(np.argmax(y_it == unique_preds) + 1) % len(unique_preds)]
                for y_it in y.numpy()
            ]
        ),
    )