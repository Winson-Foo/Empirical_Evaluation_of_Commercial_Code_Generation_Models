import os
import logging
import shutil
import requests
import zipfile
import tarfile

from .common import sha256_hash, home_directory_path

FOLDER = ".foolbox_zoo/weights"


def fetch_weights(weights_uri: str, unzip: bool = False) -> str:
    """Provides utilities to download and extract packages
    containing model weights when creating foolbox-zoo compatible
    repositories, if the weights are not part of the repository itself.

    Examples
    --------

    Download and unzip weights:

    >>> from foolbox import zoo
    >>> url = 'https://github.com/MadryLab/mnist_challenge_models/raw/master/secret.zip'  # noqa F501
    >>> weights_path = zoo.fetch_weights(url, unzip=True)

    Args:
        weights_uri: The URI to fetch the weights from.
        unzip: Should be `True` if the file to be downloaded is a zipped package.

    Returns:
        Local path where the weights have been downloaded and potentially unzipped to.
    """
    assert weights_uri is not None
    hash_digest = sha256_hash(weights_uri)
    local_path = home_directory_path(FOLDER, hash_digest)
    exists_locally = os.path.exists(local_path)

    filename = _filename_from_uri(weights_uri)
    file_path = os.path.join(local_path, filename)

    if exists_locally:
        logging.info("Weights already stored locally.")
    else:
        _download(file_path, weights_uri, local_path)

    if unzip:
        file_path = _extract(local_path, filename)

    return file_path


def _download(file_path: str, url: str, directory: str) -> None:
    """Download the weights from the given URL to the specified directory.

    Args:
        file_path: The local file path where the weights will be saved.
        url: The URL from which to download the weights.
        directory: The local directory where the weights will be saved.
    """
    logging.info("Downloading weights: %s to %s", url, file_path)
    os.makedirs(directory, exist_ok=True)
    try:
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(file_path, "wb") as f:
            shutil.copyfileobj(r.raw, f)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch weights from {url}") from e


def _extract(directory: str, file_path: str) -> str:
    """Extract the weights package to the specified directory.

    Args:
        directory: The local directory where the weights package will be extracted.
        file_path: The local file path of the weights package.

    Returns:
        The local path of the extracted folder.
    """
    extracted_folder = os.path.splitext(file_path)[0]
    extracted_folder = os.path.join(directory, extracted_folder)

    if not os.path.exists(extracted_folder):
        logging.info("Extracting weights package to %s", extracted_folder)
        os.makedirs(extracted_folder, exist_ok=True)
        if zipfile.is_zipfile(file_path):
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(extracted_folder)
        elif tarfile.is_tarfile(file_path):
            with tarfile.open(file_path, "r") as tar_ref:
                tar_ref.extractall(extracted_folder)
        else:
            raise ValueError("Unsupported weights package format")
    else:
        logging.info("Extracted folder already exists: %s", extracted_folder)

    return extracted_folder


def _filename_from_uri(url: str) -> str:
    """Extract the filename from the URL.

    Args:
        url: The URL from which to extract the filename.

    Returns:
        The extracted filename.
    """
    filename = os.path.basename(url)
    return filename.split("?")[0]