from typing import Tuple, Union, List, Any
import eagerpy as ep

import foolbox as fbn
import foolbox.attacks as fa
from foolbox.devutils import flatten
from foolbox.attacks.brendel_bethge import BrendelBethgeAttack
import pytest

from conftest import ModeAndDataAndDescription

def get_attack_id(x: Tuple[BrendelBethgeAttack, Union[int, float]]) -> str:
    return repr(x[0])

def preprocess_data(fmodel, x):
    x_normalized = (x - fmodel.bounds.lower) / (fmodel.bounds.upper - fmodel.bounds.lower)
    fmodel = fmodel.transform_bounds((0, 1))
    return fmodel, x_normalized

def run_initial_attack(fmodel, x, y):
    init_attack = fa.DatasetAttack()
    init_attack.feed(fmodel, x)
    init_advs = init_attack.run(fmodel, x, y)
    return init_advs

def run_attack(attack, fmodel, x, y, criterion, starting_points):
    advs = attack.run(fmodel, x, y, starting_points=starting_points)
    return advs

def calculate_norms(advs, x, p):
    init_norms = ep.norms.lp(flatten(advs - x), p=p, axis=-1)
    norms = ep.norms.lp(flatten(advs - x), p=p, axis=-1)
    return init_norms, norms

def test_untargeted_attack(
    request: Any,
    fmodel_and_data_ext_for_attacks: ModeAndDataAndDescription,
    attack_and_p: Tuple[fa.Attack, Union[int, float]],
) -> None:
    if request.config.option.skipslow:
        pytest.skip()

    (fmodel, x, y), real, low_dimensional_input = fmodel_and_data_ext_for_attacks

    if isinstance(x, ep.NumPyTensor):
        pytest.skip()

    if low_dimensional_input:
        pytest.skip()

    fmodel, x_normalized = preprocess_data(fmodel, x)
    init_advs = run_initial_attack(fmodel, x_normalized, y)

    attack, p = attack_and_p
    advs = run_attack(attack, fmodel, x_normalized, y, None, init_advs)

    init_norms, norms = calculate_norms(advs, x_normalized, p)

    is_smaller = norms < init_norms

    assert fbn.accuracy(fmodel, advs, y) < fbn.accuracy(fmodel, x_normalized, y)
    assert fbn.accuracy(fmodel, advs, y) <= fbn.accuracy(fmodel, init_advs, y)
    assert is_smaller.any()

def test_targeted_attack(
    request: Any,
    fmodel_and_data_ext_for_attacks: ModeAndDataAndDescription,
    attack_and_p: Tuple[fa.Attack, Union[int, float]],
) -> None:
    if request.config.option.skipslow:
        pytest.skip()

    (fmodel, x, y), real, _ = fmodel_and_data_ext_for_attacks

    if isinstance(x, ep.NumPyTensor):
        pytest.skip()

    if not real:
        pytest.skip()

    fmodel, x_normalized = preprocess_data(fmodel, x)

    logits_np = fmodel(x_normalized).numpy()
    num_classes = logits_np.shape[-1]
    y_np = logits_np.argmax(-1)

    target_classes_np = (y_np + 1) % num_classes
    for i in range(len(target_classes_np)):
        while target_classes_np[i] not in y_np:
            target_classes_np[i] = (target_classes_np[i] + 1) % num_classes
    target_classes = ep.from_numpy(y, target_classes_np)
    criterion = fbn.TargetedMisclassification(target_classes)

    init_advs = run_initial_attack(fmodel, x_normalized, criterion)

    attack, p = attack_and_p
    advs = run_attack(attack, fmodel, x_normalized, criterion, init_advs)

    init_norms, norms = calculate_norms(advs, x_normalized, p)

    is_smaller = norms < init_norms

    assert fbn.accuracy(fmodel, advs, y) < fbn.accuracy(fmodel, x_normalized, y)
    assert fbn.accuracy(fmodel, advs, target_classes) > fbn.accuracy(
        fmodel, x_normalized, target_classes
    )
    assert fbn.accuracy(fmodel, advs, target_classes) >= fbn.accuracy(
        fmodel, init_advs, target_classes
    )
    assert is_smaller.any()