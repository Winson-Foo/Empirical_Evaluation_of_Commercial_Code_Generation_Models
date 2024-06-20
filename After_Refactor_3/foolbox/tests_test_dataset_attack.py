import pytest

import foolbox as fbn

from conftest import ModeAndDataAndDescription

EPSILON_VALUES = [500.0, 1000.0]


def initialize_fmodel_and_data_ext() -> ModeAndDataAndDescription:
    fmodel_and_data_ext = fmodel_and_data_ext_for_attacks()

    (fmodel, x, y), _, _ = fmodel_and_data_ext
    x = normalize_input(x, fmodel.bounds)
    fmodel = fmodel.transform_bounds((0, 1))

    return fmodel, x, y


def normalize_input(x, bounds):
    return (x - bounds.lower) / (bounds.upper - bounds.lower)


def test_dataset_attack(fmodel_and_data_ext_for_attacks: ModeAndDataAndDescription) -> None:
    fmodel, x, y = initialize_fmodel_and_data_ext()

    run_default_dataset_attack(fmodel, x, y)
    run_custom_dataset_attack(fmodel, x, y)
    run_invalid_dataset_attack(fmodel, x, y)


def run_default_dataset_attack(fmodel, x, y):
    attack = fbn.attacks.DatasetAttack()
    attack.feed(fmodel, x)

    assert fbn.accuracy(fmodel, x, y) > 0

    advs, _, success = attack(fmodel, x, y, epsilons=None)
    assert success.shape == (len(x),)
    assert success.all()
    assert fbn.accuracy(fmodel, advs, y) == 0


def run_custom_dataset_attack(fmodel, x, y):
    attack = fbn.attacks.DatasetAttack(distance=fbn.distances.l2)
    attack.feed(fmodel, x)

    advss, _, success = attack(fmodel, x, y, epsilons=EPSILON_VALUES)
    assert success.shape == (len(EPSILON_VALUES), len(x))
    assert success.all()
    assert fbn.accuracy(fmodel, advss[0], y) == 0
    assert fbn.accuracy(fmodel, advss[1], y) == 0


def run_invalid_dataset_attack(fmodel, x, y):
    with pytest.raises(ValueError, match="unknown distance"):
        fbn.attacks.DatasetAttack()(fmodel, x, y, epsilons=EPSILON_VALUES)

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        fbn.attacks.DatasetAttack()(fmodel, x, y, epsilons=None, invalid=True)