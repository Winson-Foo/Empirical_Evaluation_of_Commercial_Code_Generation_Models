def run_attack(fmodel: fbn.Model, x: ep.Tensor, y: ep.Tensor, attack: fbn.Attack) -> Tuple[ep.Tensor, ep.Tensor, ep.Tensor]:
    assert ep.istensor(x)
    assert ep.istensor(y)

    raw, clipped, success = attack(fmodel, x, y, epsilons=1.0)
    assert ep.istensor(raw)
    assert ep.istensor(clipped)
    assert ep.istensor(success)
    assert raw.shape == x.shape
    assert clipped.shape == x.shape
    assert success.shape == (len(x),)

    return raw, clipped, success

@pytest.mark.parametrize("attack", attacks)
def test_call_one_epsilon(
    fmodel_and_data_ext_for_attacks: ModeAndDataAndDescription,
    attack: fbn.Attack,
) -> None:
    (fmodel, x, y), _, _ = fmodel_and_data_ext_for_attacks

    run_attack(fmodel, x, y, attack)

def test_get_channel_axis() -> None:
    class Model:
        data_format = None

    model = Model()

    data_formats = {
        "channels_first": 1,
        "channels_last": 2
    }

    for data_format, expected_axis in data_formats.items():
        model.data_format = data_format  # type: ignore
        assert fbn.attacks.base.get_channel_axis(model, 3) == expected_axis  # type: ignore

    model.data_format = "invalid"  # type: ignore
    with pytest.raises(ValueError):
        fbn.attacks.base.get_channel_axis(model, 3)  # type: ignore

def test_model_bounds_lower_failure(
    fmodel_and_data_ext_for_attacks: ModeAndDataAndDescription,
) -> None:
    (fmodel, x, y), _, _ = fmodel_and_data_ext_for_attacks
    attack = fbn.attacks.InversionAttack()

    with pytest.raises(AssertionError):
        attack.run(fmodel, x * 0.0 - fmodel.bounds.lower - 0.1, y)

def test_model_bounds_upper_failure(
    fmodel_and_data_ext_for_attacks: ModeAndDataAndDescription,
) -> None:
    (fmodel, x, y), _, _ = fmodel_and_data_ext_for_attacks
    attack = fbn.attacks.InversionAttack()

    with pytest.raises(AssertionError):
        attack.run(fmodel, x * 0.0 + fmodel.bounds.upper + 0.1, y)