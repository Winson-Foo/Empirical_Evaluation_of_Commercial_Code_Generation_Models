import requests
import shutil
import zipfile
import tarfile
import os

from .common import sha256_hash, home_directory_path

FOLDER = ".foolbox_zoo/weights"

def _filename_from_uri(url: str) -> str:
    filename = url.split("/")[-1]
    filename = filename.split("?")[0]
    return filename

def _download_file(file_url: str, file_path: str) -> None:
    response = requests.get(file_url, stream=True)
    if response.status_code == 200:
        with open(file_path, "wb") as file:
            response.raw.decode_content = True
            shutil.copyfileobj(response.raw, file)
    else:
        raise requests.exceptions.HTTPError(f"Failed to fetch weights from {file_url}")

def _extract_file(file_path: str, extracted_folder: str) -> None:
    if not os.path.exists(extracted_folder):
        os.makedirs(extracted_folder)
        if ".zip" in file_path:
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(extracted_folder)
        elif ".tar.gz" in file_path:
            with tarfile.open(file_path, "r") as tar_ref:
                tar_ref.extractall(extracted_folder)

def fetch_weights(weights_uri: str, unzip: bool = False) -> str:
    hash_digest = sha256_hash(weights_uri)
    local_path = home_directory_path(FOLDER, hash_digest)
    file_name = _filename_from_uri(weights_uri)
    file_path = os.path.join(local_path, file_name)

    if os.path.exists(local_path):
        return file_path

    _download_file(weights_uri, file_path)

    if unzip:
        extracted_folder = os.path.join(local_path, file_name.rsplit(".", 1)[0])
        _extract_file(file_path, extracted_folder)
        return extracted_folder

    return file_path