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


def setup_fmodel_and_data_ext_for_attacks(fmodel_and_data_ext_for_attacks):
    (fmodel, x, y), real, low_dimensional_input = fmodel_and_data_ext_for_attacks

    if isinstance(x, ep.NumPyTensor):
        return None

    if low_dimensional_input:
        return None

    x = (x - fmodel.bounds.lower) / (fmodel.bounds.upper - fmodel.bounds.lower)
    fmodel = fmodel.transform_bounds((0, 1))

    return fmodel, x, y


def run_init_attack(fmodel, x):
    init_attack = fa.DatasetAttack()
    init_attack.feed(fmodel, x)
    return init_attack.run(fmodel, x, y)


def run_attack(fmodel, x, y, attack, starting_points):
    return attack.run(fmodel, x, y, starting_points=starting_points)


def calculate_norms(advs, x, p):
    norms = ep.norms.lp(flatten(advs - x), p=p, axis=-1)
    return norms


def run_untargeted_test(fmodel, x, y, init_norms, norms):
    is_smaller = norms < init_norms

    assert fbn.accuracy(fmodel, advs, y) < fbn.accuracy(fmodel, x, y)
    assert fbn.accuracy(fmodel, advs, y) <= fbn.accuracy(fmodel, init_advs, y)
    assert is_smaller.any()


def run_targeted_test(fmodel, x, y, target_classes, init_norms, norms):
    is_smaller = norms < init_norms

    assert fbn.accuracy(fmodel, advs, y) < fbn.accuracy(fmodel, x, y)
    assert fbn.accuracy(fmodel, advs, target_classes) > fbn.accuracy(
        fmodel, x, target_classes
    )
    assert fbn.accuracy(fmodel, advs, target_classes) >= fbn.accuracy(
        fmodel, init_advs, target_classes
    )
    assert is_smaller.any()


attacks: List[Tuple[fa.Attack, Union[int, float]]] = [
    (
        fa.HopSkipJumpAttack(
            steps=1,
            constraint="linf",
            initial_gradient_eval_steps=100,
            max_gradient_eval_steps=100,
        ),
        ep.inf,
    ),
    (
        fa.HopSkipJumpAttack(
            steps=1,
            constraint="l2",
            initial_gradient_eval_steps=100,
            max_gradient_eval_steps=100,
        ),
        2,
    ),
    (
        fa.HopSkipJumpAttack(
            steps=1,
            constraint="l2",
            initial_gradient_eval_steps=100,
            max_gradient_eval_steps=100,
            stepsize_search="grid_search",
            gamma=1e5,
        ),
        2,
    ),
]


@pytest.mark.parametrize("attack_and_p", attacks, ids=get_attack_id)
def test_hsj_untargeted_attack(
    request: Any,
    fmodel_and_data_ext_for_attacks: ModeAndDataAndDescription,
    attack_and_p: Tuple[fa.HopSkipJumpAttack, Union[int, float]],
) -> None:
    if request.config.option.skipslow:
        pytest.skip()

    setup_result = setup_fmodel_and_data_ext_for_attacks(fmodel_and_data_ext_for_attacks)
    if setup_result is None:
        pytest.skip()

    fmodel, x, y = setup_result

    init_advs = run_init_attack(fmodel, x)

    attack, p = attack_and_p
    advs = run_attack(fmodel, x, y, attack, init_advs)

    init_norms = calculate_norms(init_advs, x, p)
    norms = calculate_norms(advs, x, p)

    run_untargeted_test(fmodel, x, y, init_norms, norms)


@pytest.mark.parametrize("attack_and_p", attacks, ids=get_attack_id)
def test_hsj_targeted_attack(
    request: Any,
    fmodel_and_data_ext_for_attacks: ModeAndDataAndDescription,
    attack_and_p: Tuple[fa.HopSkipJumpAttack, Union[int, float]],
) -> None:
    if request.config.option.skipslow:
        pytest.skip()

    setup_result = setup_fmodel_and_data_ext_for_attacks(fmodel_and_data_ext_for_attacks)
    if setup_result is None:
        pytest.skip()

    fmodel, x, y = setup_result

    if not real:
        pytest.skip()

    logits_np = fmodel(x).numpy()
    num_classes = logits_np.shape[-1]
    y_np = logits_np.argmax(-1)

    target_classes_np = (y_np + 1) % num_classes
    for i in range(len(target_classes_np)):
        while target_classes_np[i] not in y_np:
            target_classes_np[i] = (target_classes_np[i] + 1) % num_classes
    target_classes = ep.from_numpy(y, target_classes_np)
    criterion = fbn.TargetedMisclassification(target_classes)

    init_advs = run_init_attack(fmodel, x)

    attack, p = attack_and_p
    advs = run_attack(fmodel, x, criterion, attack, init_advs)

    init_norms = calculate_norms(init_advs, x, p)
    norms = calculate_norms(advs, x, p)

    run_targeted_test(fmodel, x, y, target_classes, init_norms, norms)