import pytest
import eagerpy as ep
import foolbox as fbn

from conftest import ModeAndDataAndDescription


def assert_tensor_shape(*tensors, shape):
    for tensor in tensors:
        assert ep.istensor(tensor)
        assert tensor.shape == shape


def test_inversion_attack_one_epsilon(
    fmodel_and_data_ext_for_attacks: ModeAndDataAndDescription,
) -> None:
    (fmodel, x, y), _, _ = fmodel_and_data_ext_for_attacks
    epsilon = 1.0

    attack = fbn.attacks.InversionAttack(distance=fbn.distances.l2)
    raw, clipped, success = attack(fmodel, x, y, epsilons=epsilon)
    
    assert_tensor_shape(raw, clipped, shape=x.shape)
    assert_tensor_shape(success, shape=(len(x),))


def test_inversion_attack_repeated_three_times_one_epsilon(
    fmodel_and_data_ext_for_attacks: ModeAndDataAndDescription,
) -> None:
    (fmodel, x, y), _, _ = fmodel_and_data_ext_for_attacks
    epsilon = 1.0

    attack = fbn.attacks.InversionAttack(distance=fbn.distances.l2).repeat(3)
    raw, clipped, success = attack(fmodel, x, y, epsilons=epsilon)
    
    assert_tensor_shape(raw, clipped, shape=x.shape)
    assert_tensor_shape(success, shape=(len(x),))


def test_l2_contrast_reduction_attack_one_epsilon(
    fmodel_and_data_ext_for_attacks: ModeAndDataAndDescription,
) -> None:
    (fmodel, x, y), _, _ = fmodel_and_data_ext_for_attacks
    epsilon = 1.0

    attack = fbn.attacks.L2ContrastReductionAttack()
    raw, clipped, success = attack(fmodel, x, y, epsilons=epsilon)
    
    assert_tensor_shape(raw, clipped, shape=x.shape)
    assert_tensor_shape(success, shape=(len(x),))


def test_l2_contrast_reduction_attack_repeated_three_times_one_epsilon(
    fmodel_and_data_ext_for_attacks: ModeAndDataAndDescription,
) -> None:
    (fmodel, x, y), _, _ = fmodel_and_data_ext_for_attacks
    epsilon = 1.0

    attack = fbn.attacks.L2ContrastReductionAttack().repeat(3)
    raw, clipped, success = attack(fmodel, x, y, epsilons=epsilon)

    assert_tensor_shape(raw, clipped, shape=x.shape)
    assert_tensor_shape(success, shape=(len(x),))


def test_get_channel_axis() -> None:
    class Model:
        data_format = None

    model = Model()
    model.data_format = "channels_first"
    assert fbn.attacks.base.get_channel_axis(model, 3) == 1
    model.data_format = "channels_last"
    assert fbn.attacks.base.get_channel_axis(model, 3) == 2
    model.data_format = "invalid"
    with pytest.raises(ValueError):
        fbn.attacks.base.get_channel_axis(model, 3)


def test_model_bounds(
    fmodel_and_data_ext_for_attacks: ModeAndDataAndDescription,
) -> None:
    (fmodel, x, y), _, _ = fmodel_and_data_ext_for_attacks
    attack = fbn.attacks.InversionAttack()

    with pytest.raises(AssertionError):
        attack.run(fmodel, x * 0.0 - fmodel.bounds.lower - 0.1, y)
    with pytest.raises(AssertionError):
        attack.run(fmodel, x * 0.0 + fmodel.bounds.upper + 0.1, y)