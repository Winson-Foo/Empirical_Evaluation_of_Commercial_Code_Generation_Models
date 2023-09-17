from typing import Tuple
import foolbox as fbn
import eagerpy as ep
import pytest


ModelAndData = Tuple[fbn.Model, ep.Tensor, ep.Tensor]


def test_accuracy(fmodel_and_data: ModelAndData) -> None:
    fmodel, x, y = fmodel_and_data
    accuracy = fbn.accuracy(fmodel, x, y)
    assert 0 <= accuracy <= 1
    assert accuracy > 0.5
    y_pred = fmodel(x).argmax(axis=-1)
    accuracy = fbn.accuracy(fmodel, x, y_pred)
    assert accuracy == 1


def test_samples(fmodel_and_data: ModelAndData, batchsize: int = 1, dataset: str = "imagenet") -> None:
    fmodel, _, _ = fmodel_and_data
    x, y = get_samples(fmodel, batchsize, dataset)
    assert len(x) == len(y) == batchsize
    assert not ep.istensor(x)
    assert not ep.istensor(y)
    if hasattr(fmodel, "data_format"):
        data_format = fmodel.data_format  # type: ignore
        x, y = get_samples(fmodel, batchsize, dataset, data_format)
        assert len(x) == len(y) == batchsize
        assert not ep.istensor(x)
        assert not ep.istensor(y)
        with pytest.raises(ValueError):
            data_format = {
                "channels_first": "channels_last",
                "channels_last": "channels_first",
            }[data_format]
            get_samples(fmodel, batchsize, dataset, data_format)
    else:
        x, y = get_samples(fmodel, batchsize, dataset, "channels_first")
        assert len(x) == len(y) == batchsize
        assert not ep.istensor(x)
        assert not ep.istensor(y)
        with pytest.raises(ValueError):
            get_samples(fmodel, batchsize, dataset)


def get_samples(fmodel: fbn.Model, batchsize: int, dataset: str, data_format: str = "channels_last") -> Tuple[ep.Tensor, ep.Tensor]:
    return fbn.samples(fmodel, dataset=dataset, batchsize=batchsize, data_format=data_format)


def test_samples_large_batch(fmodel_and_data: ModelAndData, batchsize: int = 42, dataset: str = "imagenet") -> None:
    fmodel, _, _ = fmodel_and_data
    data_format = getattr(fmodel, "data_format", "channels_first")
    with pytest.warns(UserWarning, match="only 20 samples"):
        x, y = get_samples(fmodel, batchsize, dataset, data_format)
    assert len(x) == len(y) == batchsize
    assert not ep.istensor(x)
    assert not ep.istensor(y)