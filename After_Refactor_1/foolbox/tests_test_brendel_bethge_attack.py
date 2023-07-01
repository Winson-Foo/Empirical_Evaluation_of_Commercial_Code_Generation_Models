from typing import Tuple, Union, List, Any
import eagerpy as ep
import foolbox as fbn
import foolbox.attacks as fa
import pytest

from conftest import ModeAndDataAndDescription

def get_attack_id(attack: fa.Attack, p: Union[int, float]) -> str:
    return repr(attack)

ATTACKS = [
    (fa.L0BrendelBethgeAttack(steps=20), 0),
    (fa.L1BrendelBethgeAttack(steps=20), 1),
    (fa.L2BrendelBethgeAttack(steps=20), 2),
    (fa.LinfinityBrendelBethgeAttack(steps=20), ep.inf),
]

def transform_input_bounds(fmodel: fbn.Model, x: ep.Tensor) -> ep.Tensor:
    return (x - fmodel.bounds.lower) / (fmodel.bounds.upper - fmodel.bounds.lower)

def run_initial_attack(fmodel: fbn.Model, x: ep.Tensor) -> ep.Tensor:
    init_attack = fa.DatasetAttack()
    init_attack.feed(fmodel, x)
    return init_attack.run(fmodel, x)

def run_attack(fmodel: fbn.Model, x: ep.Tensor, y: ep.Tensor, attack: fa.Attack, starting_points: ep.Tensor) -> ep.Tensor:
    return attack.run(fmodel, x, y, starting_points=starting_points)

def calculate_lp_norms(advs: ep.Tensor, x: ep.Tensor, p: Union[int, float]) -> ep.Tensor:
    return ep.norms.lp(flatten(advs - x), p=p, axis=-1)

def is_norm_smaller(norms: ep.Tensor, init_norms: ep.Tensor) -> bool:
    return norms < init_norms

@pytest.mark.parametrize("attack_and_p", ATTACKS, ids=get_attack_id)
def test_brendel_bethge_untargeted_attack(
    request: Any,
    fmodel_and_data_ext_for_attacks: ModeAndDataAndDescription,
    attack_and_p: Tuple[fa.Attack, Union[int, float]],
) -> None:
    if request.config.option.skipslow:
        pytest.skip()

    (fmodel, x, y), real, low_dimensional_input = fmodel_and_data_ext_for_attacks

    x = transform_input_bounds(fmodel, x)
    fmodel = fmodel.transform_bounds((0, 1))

    init_advs = run_initial_attack(fmodel, x)

    attack, p = attack_and_p
    advs = run_attack(fmodel, x, y, attack, starting_points=init_advs)

    init_norms = calculate_lp_norms(init_advs, x, p)
    norms = calculate_lp_norms(advs, x, p)

    is_smaller = is_norm_smaller(norms, init_norms)

    assert fbn.accuracy(fmodel, advs, y) < fbn.accuracy(fmodel, x, y)
    assert fbn.accuracy(fmodel, advs, y) <= fbn.accuracy(fmodel, init_advs, y)
    assert is_smaller.any()