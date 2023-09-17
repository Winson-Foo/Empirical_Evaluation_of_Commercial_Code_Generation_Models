import hashlib
import os


def calculate_sha256_hash(git_uri: str) -> str:
    """Calculates the SHA256 hash of the given string."""
    m = hashlib.sha256()
    m.update(git_uri.encode())
    return m.hexdigest()


def get_home_directory_path(folder: str, hash_digest: str) -> str:
    """Returns the path to the folder in the user's home directory using the provided hash digest."""
    home_directory = os.path.expanduser("~")
    return os.path.join(home_directory, folder, hash_digest)