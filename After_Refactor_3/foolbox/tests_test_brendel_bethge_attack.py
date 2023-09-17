from typing import Tuple, Union, List, Any
import eagerpy as ep
import foolbox as fbn
import pytest

def get_attack_id(attack: fbn.attacks.Attack, p: Union[int, float]) -> str:
    return repr(attack)

def run_brendel_bethge_attack(fmodel, x, y, attack, p):
    init_attack = fbn.attacks.DatasetAttack()
    init_attack.feed(fmodel, x)
    init_advs = init_attack.run(fmodel, x, y)

    advs = attack.run(fmodel, x, y, starting_points=init_advs)

    init_norms = ep.norms.lp(flatten(init_advs - x), p=p, axis=-1)
    norms = ep.norms.lp(flatten(advs - x), p=p, axis=-1)

    is_smaller = norms < init_norms

    assert fbn.accuracy(fmodel, advs, y) < fbn.accuracy(fmodel, x, y)
    assert fbn.accuracy(fmodel, advs, y) <= fbn.accuracy(fmodel, init_advs, y)
    assert is_smaller.any()

def test_brendel_bethge_untargeted_attack(
    request: Any,
    fmodel_and_data_ext_for_attacks: Tuple,
    attack_and_p: Tuple[fbn.attacks.Attack, Union[int, float]],
) -> None:
    if request.config.option.skipslow:
        pytest.skip()

    (fmodel, x, y), real, low_dimensional_input = fmodel_and_data_ext_for_attacks

    if isinstance(x, ep.NumPyTensor):
        pytest.skip()

    if low_dimensional_input:
        pytest.skip()

    x = (x - fmodel.bounds.lower) / (fmodel.bounds.upper - fmodel.bounds.lower)
    fmodel = fmodel.transform_bounds((0, 1))

    attack, p = attack_and_p
    run_brendel_bethge_attack(fmodel, x, y, attack, p)