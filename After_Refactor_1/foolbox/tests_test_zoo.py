import sys
import pytest

import foolbox as fbn


@pytest.fixture(autouse=True)
def reload_foolbox_model_module() -> None:
    # reload foolbox_model from scratch for every run
    # to ensure atomic tests without side effects
    module_names = ["foolbox_model", "model"]
    for module_name in module_names:
        if module_name in sys.modules:
            del sys.modules[module_name]


urls = [
    "https://github.com/jonasrauber/foolbox-tensorflow-keras-applications",
    "git@github.com:jonasrauber/foolbox-tensorflow-keras-applications.git",
]


@pytest.mark.parametrize("url", urls)
def test_loading_valid_model(request: Any, url: str) -> None:
    backend = request.config.option.backend
    if backend != "tensorflow":
        pytest.skip()

    # download model
    try:
        fmodel = fbn.zoo.get_model(url, name="MobileNetV2", overwrite=True)
    except fbn.zoo.GitCloneError:
        pytest.skip()

    # download again (test overwriting)
    try:
        fmodel = fbn.zoo.get_model(url, name="MobileNetV2", overwrite=True)
    except fbn.zoo.GitCloneError:
        pytest.skip()

    # create a dummy image
    x, y = fbn.samples(fmodel, dataset="imagenet", batchsize=16)

    # run the model
    assert fbn.accuracy(fmodel, x, y) > 0.9


def test_loading_invalid_url(request: Any) -> None:
    url = "https://github.com/jonasrauber/invalid-url"
    with pytest.raises(fbn.zoo.GitCloneError):
        fbn.zoo.get_model(url, name="MobileNetV2", overwrite=True)


def test_non_default_module_throws_error() -> None:
    with pytest.raises(ValueError):
        fbn.zoo.model_loader.get(key="other")