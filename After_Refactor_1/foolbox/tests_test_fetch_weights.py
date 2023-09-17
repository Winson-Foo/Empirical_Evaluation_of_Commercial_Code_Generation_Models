import os
import pytest
import shutil
import hashlib
import requests

from foolbox.zoo import fetch_weights
from foolbox.zoo.common import home_directory_path

WEIGHTS_FOLDER = "weights"
SERVER_URL = "http://localhost:8080"

@pytest.fixture(autouse=True)
def clear_weights_folder():
    weights_folder = home_directory_path(WEIGHTS_FOLDER)
    if os.path.exists(weights_folder):
        shutil.rmtree(weights_folder)
    yield
    if os.path.exists(weights_folder):
        shutil.rmtree(weights_folder)

@pytest.mark.parametrize("unzip", [True, False])
def test_fetch_weights(unzip):
    weights_uri = f"{SERVER_URL}/weights.zip"
    raw_body = _random_body(zipped=unzip)

    # mock server
    with responses.RequestsMock() as rsps:
        rsps.add(responses.GET, weights_uri, body=raw_body, status=200, stream=True)

        expected_path = _expected_path(weights_uri)

        file_path = fetch_weights(weights_uri, unzip=unzip)

        exists_locally = os.path.exists(expected_path)
        assert exists_locally
        assert expected_path in file_path

def test_fetch_weights_returns_404():
    weights_uri = f"{SERVER_URL}/weights.zip"

    # mock server
    with responses.RequestsMock() as rsps:
        rsps.add(responses.GET, weights_uri, status=404)

        expected_path = _expected_path(weights_uri)

        with pytest.raises(RuntimeError):
            fetch_weights(weights_uri, unzip=False)

def _random_body(zipped=False):
    if zipped:
        data = io.BytesIO()
        with zipfile.ZipFile(data, mode="w") as z:
            z.writestr("test.txt", "no real weights in here :)")
        data.seek(0)
        return data.getvalue()
    else:
        raw_body = os.urandom(1024)
        return raw_body

def _expected_path(weights_uri):
    hash_digest = hashlib.sha256(weights_uri.encode()).hexdigest()
    local_path = home_directory_path(WEIGHTS_FOLDER, hash_digest)
    return local_path