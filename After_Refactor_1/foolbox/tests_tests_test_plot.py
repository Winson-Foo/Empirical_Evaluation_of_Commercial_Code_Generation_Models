import pytest
import eagerpy as ep
import foolbox as fbn


def test_plot(dummy: ep.Tensor) -> None:
    dummy_shape = (10, 3, 32, 32)
    dummy_shape_single_channel = (10, 32, 32, 1)

    test_cases = [
        (dummy_shape, {}),
        (dummy_shape, {"n": 3}),
        (dummy_shape, {"n": 3, "data_format": "channels_first"}),
        (dummy_shape, {"nrows": 4}),
        (dummy_shape, {"ncols": 3}),
        (dummy_shape, {"nrows": 2, "ncols": 6}),
        (dummy_shape, {"nrows": 2, "ncols": 4}),
        (dummy_shape_single_channel, {}),
    ]

    for image_shape, kwargs in test_cases:
        images = ep.zeros(dummy, image_shape)
        fbn.plot.images(images, **kwargs)

    invalid_test_cases = [
        (dummy_shape, {"data_format": "foo"}),
        (dummy_shape, {"shape": (10, 3, 3, 3)}),
        (dummy_shape, {"shape": (10, 1, 1, 1)}),
        (dummy_shape, {"shape": (10, 32, 32)}),
        (dummy_shape, {"shape": (10, 3, 32, 32), "data_format": "foo"}),
    ]

    for image_shape, kwargs in invalid_test_cases:
        images = ep.zeros(dummy, image_shape)
        with pytest.raises(ValueError):
            fbn.plot.images(images, **kwargs)