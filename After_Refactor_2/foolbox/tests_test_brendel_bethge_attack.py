from typing import Tuple, Union, List, Any
import eagerpy as ep
import foolbox as fbn
import foolbox.attacks as fa
from foolbox.devutils import flatten
import pytest

from conftest import ModeAndDataAndDescription


def get_attack_id(attack: fa.Attack, p: Union[int, float]) -> str:
    return repr(attack)


attacks: List[Tuple[fa.Attack, Union[int, float]]] = [
    (fa.L0BrendelBethgeAttack(steps=20), 0),
    (fa.L1BrendelBethgeAttack(steps=20), 1),
    (fa.L2BrendelBethgeAttack(steps=20), 2),
    (fa.LinfinityBrendelBethgeAttack(steps=20), ep.inf),
]


def preprocess_data(fmodel: Any, x: Any, y: Any) -> Tuple[Any, Any, bool]:
    low_dimensional_input = False

    if isinstance(x, ep.NumPyTensor):
        return None, None, low_dimensional_input

    if low_dimensional_input:
        return None, None, low_dimensional_input

    x = (x - fmodel.bounds.lower) / (fmodel.bounds.upper - fmodel.bounds.lower)
    fmodel = fmodel.transform_bounds((0, 1))

    return fmodel, x, low_dimensional_input


def run_brendel_bethge_attack(
    fmodel: Any,
    x: Any,
    y: Any,
    attack: fa.Attack,
    init_advs: Any,
    p: Union[int, float]
) -> Tuple[bool, Any]:
    advs = attack.run(fmodel, x, y, starting_points=init_advs)

    init_norms = ep.norms.lp(flatten(init_advs - x), p=p, axis=-1)
    norms = ep.norms.lp(flatten(advs - x), p=p, axis=-1)

    is_smaller = norms < init_norms

    return is_smaller.any(), advs


@pytest.mark.parametrize("attack_and_p", attacks, ids=get_attack_id)
def test_brendel_bethge_untargeted_attack(
    request: Any,
    fmodel_and_data_ext_for_attacks: ModeAndDataAndDescription,
    attack_and_p: Tuple[fa.Attack, Union[int, float]],
) -> None:
    if request.config.option.skipslow:
        pytest.skip()

    (fmodel, x, y), real, low_dimensional_input = fmodel_and_data_ext_for_attacks

    fmodel, x, low_dimensional_input = preprocess_data(fmodel, x, y)
    if fmodel is None or x is None:
        pytest.skip()
    
    init_attack = fa.DatasetAttack()
    init_attack.feed(fmodel, x)
    init_advs = init_attack.run(fmodel, x, y)

    attack, p = attack_and_p
    is_smaller, advs = run_brendel_bethge_attack(fmodel, x, y, attack, init_advs, p)

    assert fbn.accuracy(fmodel, advs, y) < fbn.accuracy(fmodel, x, y)
    assert fbn.accuracy(fmodel, advs, y) <= fbn.accuracy(fmodel, init_advs, y)
    assert is_smaller