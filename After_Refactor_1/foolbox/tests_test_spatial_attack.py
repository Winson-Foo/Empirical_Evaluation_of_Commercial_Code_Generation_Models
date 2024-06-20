from typing import List, Tuple
import pytest
import foolbox as fbn
import foolbox.attacks as fa

from conftest import ModeAndDataAndDescription


def get_attack_id(attack: fbn.Attack) -> str:
    return repr(attack)


def preprocess_data(fmodel: fbn.Model, x: fbn.Tensor, y: fbn.Tensor) -> Tuple[fbn.Tensor, bool]:
    x_normalized = (x - fmodel.bounds.lower) / (fmodel.bounds.upper - fmodel.bounds.lower)
    fmodel_transformed = fmodel.transform_bounds((0, 1))
    acc = fbn.accuracy(fmodel_transformed, x_normalized, y)
    return acc, x_normalized


def test_spatial_attacks(
    fmodel_and_data_ext_for_attacks: ModeAndDataAndDescription,
    attack_grad_real: Tuple[fbn.Attack, bool],
) -> None:
    attack, repeated = attack_grad_real
    if repeated:
        attack = attack.repeat(2)
    (fmodel, x, y), real, _ = fmodel_and_data_ext_for_attacks
    if not real:
        pytest.skip()

    acc, x_normalized = preprocess_data(fmodel, x, y)
    assert acc > 0

    advs, _, _ = attack(fmodel, x_normalized, y)
    assert fbn.accuracy(fmodel, advs, y) < acc