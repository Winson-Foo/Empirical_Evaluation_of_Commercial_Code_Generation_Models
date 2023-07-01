from typing import Tuple, Union, List
import eagerpy as ep
import foolbox as fbn
import foolbox.attacks as fa
from foolbox.attacks.fast_minimum_norm import FMNAttackLp
import pytest
import numpy as np
from conftest import ModeAndDataAndDescription


def get_attack_id(attack: FMNAttackLp, p: Union[int, float]) -> str:
    return repr(attack)


def preprocess_data(
    fmodel: fbn.Model,
    x: ep.Tensor,
    y: ep.Tensor
) -> Tuple[ep.Tensor, fbn.Model]:
    if isinstance(x, ep.NumPyTensor):
        pytest.skip()

    x = (x - fmodel.bounds.lower) / (fmodel.bounds.upper - fmodel.bounds.lower)
    fmodel = fmodel.transform_bounds((0, 1))

    return x, fmodel


def run_untargeted_attack(
    fmodel: fbn.Model,
    x: ep.Tensor,
    y: ep.Tensor,
    attack: fa.Attack,
    init_advs: ep.Tensor,
    p: Union[int, float]
) -> ep.Tensor:
    advs = attack.run(fmodel, x, y, starting_points=init_advs)

    init_norms = ep.norms.lp(flatten(init_advs - x), p=p, axis=-1)
    norms = ep.norms.lp(flatten(advs - x), p=p, axis=-1)

    is_smaller = norms < init_norms

    assert fbn.accuracy(fmodel, advs, y) < fbn.accuracy(fmodel, x, y)
    assert fbn.accuracy(fmodel, advs, y) <= fbn.accuracy(fmodel, init_advs, y)
    assert is_smaller.any()

    return advs


def run_targeted_attack(
    fmodel: fbn.Model,
    x: ep.Tensor,
    y: ep.Tensor,
    attack: fa.Attack,
    init_advs: ep.Tensor,
    p: Union[int, float],
    target_classes: ep.Tensor
) -> ep.Tensor:
    advs = attack.run(fmodel, x, target_classes, starting_points=init_advs)

    init_norms = ep.norms.lp(flatten(init_advs - x), p=p, axis=-1)
    norms = ep.norms.lp(flatten(advs - x), p=p, axis=-1)

    is_smaller = norms < init_norms

    assert fbn.accuracy(fmodel, advs, target_classes) == 1.0
    assert is_smaller.any()

    return advs


attacks: List[Tuple[fa.Attack, Union[int, float]]] = [
    (fa.L0FMNAttack(steps=20), 0),
    (fa.L1FMNAttack(steps=20), 1),
    (fa.L2FMNAttack(steps=20), 2),
    (fa.LInfFMNAttack(steps=20), ep.inf),
    (fa.LInfFMNAttack(steps=20, min_stepsize=1.0 / 100), ep.inf),
]


@pytest.mark.parametrize("attack_and_p", attacks, ids=get_attack_id)
def test_fast_minimum_norm_untargeted_attack(
    fmodel_and_data_ext_for_attacks: ModeAndDataAndDescription,
    attack_and_p: Tuple[fa.Attack, Union[int, float]],
) -> None:
    (fmodel, x, y), real, low_dimensional_input = fmodel_and_data_ext_for_attacks
    x, fmodel = preprocess_data(fmodel, x, y)

    init_attack = fa.DatasetAttack()
    init_attack.feed(fmodel, x)
    init_advs = init_attack.run(fmodel, x, y)

    attack, p = attack_and_p
    run_untargeted_attack(fmodel, x, y, attack, init_advs, p)


@pytest.mark.parametrize("attack_and_p", attacks, ids=get_attack_id)
def test_fast_minimum_norm_targeted_attack(
    fmodel_and_data_ext_for_attacks: ModeAndDataAndDescription,
    attack_and_p: Tuple[fa.Attack, Union[int, float]],
) -> None:
    (fmodel, x, y), real, low_dimensional_input = fmodel_and_data_ext_for_attacks
    x, fmodel = preprocess_data(fmodel, x, y)

    unique_preds = np.unique(fmodel(x).argmax(-1).numpy())
    target_classes = ep.from_numpy(
        y,
        np.array(
            [
                unique_preds[(np.argmax(y_it == unique_preds) + 1) % len(unique_preds)]
                for y_it in y.numpy()
            ]
        ),
    )
    criterion = fbn.TargetedMisclassification(target_classes)
    adv_before_attack = criterion(x, fmodel(x))
    assert not adv_before_attack.all()

    init_attack = fa.DatasetAttack()
    init_attack.feed(fmodel, x)
    init_advs = init_attack.run(fmodel, x, criterion)

    attack, p = attack_and_p
    run_targeted_attack(fmodel, x, y, attack, init_advs, p, target_classes)