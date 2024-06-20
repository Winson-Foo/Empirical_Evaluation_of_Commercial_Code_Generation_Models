import pytest
import numpy as np
import foolbox as fbn


urls = [
    "https://github.com/jonasrauber/foolbox-tensorflow-keras-applications",
    "git@github.com:jonasrauber/foolbox-tensorflow-keras-applications.git",
]


@pytest.mark.parametrize("url", urls)
def test_loading_model(request, url):
    backend = request.config.option.backend
    if backend != "tensorflow":
        pytest.skip()

    model = "MobileNetV2"
    overwrite = True

    try:
        fmodel = fbn.zoo.get_model(url, name=model, overwrite=overwrite)
    except fbn.zoo.GitCloneError:
        pytest.skip()

    try:
        fmodel = fbn.zoo.get_model(url, name=model, overwrite=overwrite)
    except fbn.zoo.GitCloneError:
        pytest.skip()

    x, y = fbn.samples(fmodel, dataset="imagenet", batchsize=16)

    assert fbn.accuracy(fmodel, x, y) > 0.9


def test_loading_invalid_model(request):
    url = "https://github.com/jonasrauber/invalid-url"
    with pytest.raises(fbn.zoo.GitCloneError):
        fbn.zoo.get_model(url, name="MobileNetV2", overwrite=True)


def test_non_default_module_throws_error():
    with pytest.raises(ValueError):
        fbn.zoo.ModelLoader.get(key="other")