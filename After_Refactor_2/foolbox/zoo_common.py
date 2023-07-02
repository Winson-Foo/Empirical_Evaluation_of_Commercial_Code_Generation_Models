import hashlib
import os


def calculate_sha256_hash(git_uri: str) -> str:
    """
    Calculate the SHA256 hash digest for a given Git URI.

    Args:
        git_uri (str): The Git URI to calculate the hash for.

    Returns:
        str: The resulting SHA256 hash digest.
    """
    hash_object = hashlib.sha256()
    hash_object.update(git_uri.encode())
    return hash_object.hexdigest()


def get_home_directory_path(folder: str, hash_digest: str) -> str:
    """
    Get the path to the home directory based on a given folder and hash digest.

    Args:
        folder (str): The name of the folder.
        hash_digest (str): The hash digest to include in the path.

    Returns:
        str: The path to the home directory with the specified folder and hash digest.
    """
    home_directory = os.path.expanduser("~")
    return os.path.join(home_directory, folder, hash_digest)