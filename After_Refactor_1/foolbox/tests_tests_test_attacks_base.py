import pytest
import eagerpy as ep
import foolbox as fbn

from conftest import ModeAndDataAndDescription


def attack_test_data():
    attacks = [
        fbn.attacks.InversionAttack(distance=fbn.distances.l2),
        fbn.attacks.InversionAttack(distance=fbn.distances.l2).repeat(3),
        fbn.attacks.L2ContrastReductionAttack(),
        fbn.attacks.L2ContrastReductionAttack().repeat(3),
    ]
    return attacks


@pytest.mark.parametrize("attack", attack_test_data())
def test_call_one_epsilon(fmodel_and_data_ext: ModeAndDataAndDescription, attack: fbn.Attack) -> None:
    fmodel, x, y = fmodel_and_data_ext

    assert ep.istensor(x)
    assert ep.istensor(y)

    raw, clipped, success = attack(fmodel, x, y, epsilons=1.0)
    assert ep.istensor(raw)
    assert ep.istensor(clipped)
    assert ep.istensor(success)
    assert raw.shape == x.shape
    assert clipped.shape == x.shape
    assert success.shape == (len(x),)


def test_get_channel_axis():
    class Model:
        data_format = None

    model = Model()
    model.data_format = "channels_first"
    assert fbn.attacks.base.get_channel_axis(model, 3) == 1
    model.data_format = "channels_last"
    assert fbn.attacks.base.get_channel_axis(model, 3) == 2
    model.data_format = "invalid"
    with pytest.raises(ValueError):
        assert fbn.attacks.base.get_channel_axis(model, 3)


def test_model_bounds(fmodel_and_data_ext: ModeAndDataAndDescription) -> None:
    fmodel, x, y = fmodel_and_data_ext
    attack = fbn.attacks.InversionAttack()

    with pytest.raises(AssertionError):
        attack.run(fmodel, x * 0.0 - fmodel.bounds.lower - 0.1, y)
    with pytest.raises(AssertionError):
        attack.run(fmodel, x * 0.0 + fmodel.bounds.upper + 0.1, y)