import os
import pytest
import shutil
import responses
import io
import zipfile
from foolbox.zoo.weights_fetcher import FOLDER
from foolbox.zoo.common import home_directory_path, sha256_hash


def fetch_weights(weights_uri: str, unzip: bool = False) -> str:
    responses.add(responses.GET, weights_uri, status=200, stream=True)
    expected_path = _expected_path(weights_uri)

    if os.path.exists(expected_path):
        shutil.rmtree(expected_path)  # make sure path does not exist already

    file_path = _download_weights(weights_uri)

    if unzip:
        file_path = _unzip_weights(file_path)

    exists_locally = os.path.exists(expected_path)
    assert exists_locally
    assert expected_path in file_path

    return file_path


def _download_weights(weights_uri: str) -> str:
    response = responses.requests.get(weights_uri)
    response.raise_for_status()

    expected_path = _expected_path(weights_uri)
    os.makedirs(os.path.dirname(expected_path), exist_ok=True)

    with open(expected_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    return expected_path


def _unzip_weights(file_path: str) -> str:
    unzip_path = file_path[:-4]  # remove .zip extension from file path
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(unzip_path)

    os.remove(file_path)  # remove the .zip file after extraction
    return unzip_path


@responses.activate
def test_fetch_weights_unzipped() -> None:
    weights_uri = "http://localhost:8080/weights.zip"
    raw_body = _random_body(zipped=False)

    # mock server
    responses.add(responses.GET, weights_uri, body=raw_body, status=200, stream=True)

    expected_path = _expected_path(weights_uri)

    if os.path.exists(expected_path):
        shutil.rmtree(expected_path)  # make sure path does not exist already

    file_path = fetch_weights(weights_uri)

    exists_locally = os.path.exists(expected_path)
    assert exists_locally
    assert expected_path in file_path


@responses.activate
def test_fetch_weights_zipped() -> None:
    weights_uri = "http://localhost:8080/weights.zip"

    # mock server
    raw_body = _random_body(zipped=True)
    responses.add(
        responses.GET,
        weights_uri,
        body=raw_body,
        status=200,
        stream=True,
        content_type="application/zip",
        headers={"Accept-Encoding": "gzip, deflate"},
    )

    expected_path = _expected_path(weights_uri)

    if os.path.exists(expected_path):
        shutil.rmtree(expected_path)  # make sure path does not exist already

    file_path = fetch_weights(weights_uri, unzip=True)

    exists_locally = os.path.exists(expected_path)
    assert exists_locally
    assert expected_path in file_path


@responses.activate
def test_fetch_weights_returns_404() -> None:
    weights_uri = "http://down:8080/weights.zip"

    # mock server
    responses.add(responses.GET, weights_uri, status=404)

    expected_path = _expected_path(weights_uri)

    if os.path.exists(expected_path):
        shutil.rmtree(expected_path)  # make sure path does not exist already

    with pytest.raises(RuntimeError):
        fetch_weights(weights_uri, unzip=False)


def _random_body(zipped: bool = False) -> bytes:
    if zipped:
        data = io.BytesIO()
        with zipfile.ZipFile(data, mode="w") as z:
            z.writestr("test.txt", "no real weights in here :)")
        data.seek(0)
        return data.getvalue()
    else:
        raw_body = os.urandom(1024)
        return raw_body


def _expected_path(weights_uri: str) -> str:
    hash_digest = sha256_hash(weights_uri)
    local_path = home_directory_path(FOLDER, hash_digest)
    return local_path