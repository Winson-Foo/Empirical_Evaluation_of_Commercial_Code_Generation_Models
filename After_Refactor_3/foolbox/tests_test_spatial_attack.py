from typing import List, Tuple
import pytest
import foolbox as fbn
import foolbox.attacks as fa

from conftest import ModeAndDataAndDescription


def get_attack_id(attack: fbn.Attack, repeated: bool) -> str:
    return repr(attack)


def preprocess_data(fmodel: fbn.models.Model, x: fbn.Tensor, y: fbn.Tensor) -> Tuple[fbn.Tensor, float]:
    # Normalize the input data
    x_normalized = (x - fmodel.bounds.lower) / (fmodel.bounds.upper - fmodel.bounds.lower)
    
    # Transform the bounds
    fmodel = fmodel.transform_bounds((0, 1))
    
    # Calculate the initial accuracy
    initial_acc = fbn.accuracy(fmodel, x_normalized, y)
    
    return x_normalized, initial_acc


def run_attack(attack: fbn.Attack, fmodel: fbn.models.Model, x: fbn.Tensor, y: fbn.Tensor) -> Tuple[float, fbn.Tensor]:
    # Generate adversarial examples
    advs, _, _ = attack(fmodel, x, y)  # type: ignore
    
    # Calculate the accuracy on adversarial examples
    adv_acc = fbn.accuracy(fmodel, advs, y)
    
    return adv_acc, advs


def test_spatial_attacks(
    fmodel_and_data_ext_for_attacks: ModeAndDataAndDescription,
    attack_grad_real: Tuple[fbn.Attack, bool]
) -> None:
    attack, repeated = attack_grad_real
    
    # Repeat the attack if necessary
    if repeated:
        attack = attack.repeat(2)
    
    # Prepare the data
    (fmodel, x, y), real, _ = fmodel_and_data_ext_for_attacks
    
    # Skip the test if not real data
    if not real:
        pytest.skip()
    
    # Preprocess the data
    x_normalized, initial_acc = preprocess_data(fmodel, x, y)
    
    # Assert that initial accuracy is greater than 0
    assert initial_acc > 0
    
    # Run the attack
    adv_acc, _ = run_attack(attack, fmodel, x_normalized, y)
    
    # Assert that accuracy on adversarial examples is less than initial accuracy
    assert adv_acc < initial_acc