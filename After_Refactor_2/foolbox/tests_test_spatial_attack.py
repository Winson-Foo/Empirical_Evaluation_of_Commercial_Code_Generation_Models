from typing import List, Tuple
import pytest
import foolbox as fbn
import foolbox.attacks as fa

from conftest import ModeAndDataAndDescription

def get_attack_id(attack: fbn.Attack) -> str:
    return repr(attack)

def normalize_data(fmodel: fbn.models.Model, x: fbn.Tensor, bounds: Tuple[float, float]) -> fbn.Tensor:
    lower, upper = bounds
    return (x - lower) / (upper - lower)

def transform_bounds(fmodel: fbn.models.Model, bounds: Tuple[float, float]) -> fbn.models.Model:
    return fmodel.transform_bounds((0, 1))

def test_spatial_attacks(
    fmodel_and_data_ext_for_attacks: ModeAndDataAndDescription,
    attack_grad_real: Tuple[fbn.Attack, bool]
) -> None:
    attack, repeated = attack_grad_real
    if repeated:
        attack = attack.repeat(2)

    (fmodel, x, y), real, _ = fmodel_and_data_ext_for_attacks
    if not real:
        pytest.skip()
        
    normalized_x = normalize_data(fmodel, x, fmodel.bounds)
    transformed_fmodel = transform_bounds(fmodel, fmodel.bounds)
    
    original_accuracy = fbn.accuracy(transformed_fmodel, normalized_x, y)
    assert original_accuracy > 0
    
    advs, _, _ = attack(transformed_fmodel, normalized_x, y)
    assert fbn.accuracy(transformed_fmodel, advs, y) < original_accuracy