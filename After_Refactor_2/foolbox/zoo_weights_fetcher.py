import os
import shutil
import tarfile
import zipfile
import logging
import requests
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

    filename = get_filename_from_uri(weights_uri)
    file_path = os.path.join(local_path, filename)

    if exists_locally:
        logging.info("Weights already stored locally.")
    else:
        download_weights(file_path, weights_uri, local_path)

    if unzip:
        file_path = extract_file(local_path, filename)

    return file_path


def get_filename_from_uri(uri: str) -> str:
    filename = uri.split("/")[-1]
    filename = filename.split("?")[0]
    return filename


def download_weights(file_path: str, url: str, directory: str) -> None:
    logger = logging.getLogger(__name__)
    logger.info("Downloading weights: %s to %s", url, file_path)

    if not os.path.exists(directory):
        os.makedirs(directory)

    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(file_path, "wb") as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, f)
    else:
        raise RuntimeError("Failed to fetch weights from %s", url)


def extract_file(directory: str, filename: str) -> str:
    file_path = os.path.join(directory, filename)
    extracted_folder = filename.rsplit(".", 1)[0]
    extracted_folder = os.path.join(directory, extracted_folder)

    if not os.path.exists(extracted_folder):
        logging.info("Extracting weights package to %s", extracted_folder)
        os.makedirs(extracted_folder)

        if ".zip" in file_path:
            zip_ref = zipfile.ZipFile(file_path, "r")
            zip_ref.extractall(extracted_folder)
            zip_ref.close()

        elif ".tar.gz" in file_path:
            tar_ref = tarfile.TarFile.open(file_path, "r")
            tar_ref.extractall(extracted_folder)
            tar_ref.close()
    else:
        logging.info(
            "Extracted folder already exists: %s", extracted_folder
        )

    return extracted_folder