import hashlib
import os


def calculate_sha256_hash(git_uri: str) -> str:
    """
    Calculates the SHA256 hash of the given git URI.

    Args:
        git_uri (str): The git URI to calculate the hash for.

    Returns:
        str: The SHA256 hash digest.
    """
    m = hashlib.sha256()
    m.update(git_uri.encode())
    return m.hexdigest()


def get_home_directory_path(folder: str, hash_digest: str) -> str:
    """
    Generates the home directory path based on a folder and hash digest.

    Args:
        folder (str): The folder name.
        hash_digest (str): The hash digest.

    Returns:
        str: The complete home directory path.
    """
    home = os.path.expanduser("~")
    return os.path.join(home, folder, hash_digest)