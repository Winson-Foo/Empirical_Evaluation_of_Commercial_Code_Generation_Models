from typing import Any
import sys
import pytest

import foolbox as fbn
from foolbox.zoo.model_loader import ModelLoader


@pytest.fixture(autouse=True)
def unload_foolbox_model_module() -> None:
    modules_to_unload = ["foolbox_model", "model"]
    for module_name in modules_to_unload:
        if module_name in sys.modules:
            del sys.modules[module_name]


@pytest.mark.parametrize("url", [
    "https://github.com/jonasrauber/foolbox-tensorflow-keras-applications",
    "git@github.com:jonasrauber/foolbox-tensorflow-keras-applications.git",
])
def test_loading_model(request: Any, url: str) -> None:
    backend = request.config.option.backend
    if backend != "tensorflow":
        pytest.skip()

    try:
        fmodel = fbn.zoo.get_model(url, name="MobileNetV2", overwrite=True)
        fmodel_2 = fbn.zoo.get_model(url, name="MobileNetV2", overwrite=True)  # Test overwriting
        x, y = fbn.samples(fmodel, dataset="imagenet", batchsize=16)
        assert fbn.accuracy(fmodel, x, y) > 0.9
    except fbn.zoo.GitCloneError:
        pytest.skip()


def test_loading_invalid_model(request: Any) -> None:
    url = "https://github.com/jonasrauber/invalid-url"
    with pytest.raises(fbn.zoo.GitCloneError):
        fbn.zoo.get_model(url, name="MobileNetV2", overwrite=True)


def test_non_default_module_throws_error() -> None:
    with pytest.raises(ValueError):
        ModelLoader.get(key="other")