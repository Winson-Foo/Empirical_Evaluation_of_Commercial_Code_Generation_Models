import shutil
from git import Repo
from .common import sha256_hash, home_directory_path


FOLDER = ".foolbox_zoo"


def clone_repo(git_uri: str, overwrite: bool = False) -> str:
    """Clones a remote git repository to a local path.

    Args:
        git_uri: The URI to the git repository to be cloned.
        overwrite: Whether or not to overwrite the local path.

    Returns:
        The generated local path where the repository has been cloned to.
    """
    hash_digest = sha256_hash(git_uri)
    local_path = home_directory_path(FOLDER, hash_digest)
    exists_locally = shutil.rmtree(local_path, ignore_errors=True) if overwrite else os.path.exists(local_path)

    if not exists_locally:
        _clone_repo(git_uri, local_path)
    else:
        print("Git repository already exists locally.")

    return local_path


def _clone_repo(git_uri: str, local_path: str) -> None:
    print(f"Cloning repo {git_uri} to {local_path}")
    try:
        Repo.clone_from(git_uri, local_path)
    except Exception as e:
        print("Failed to clone repository")
        raise RuntimeError("Failed to clone repository")

    print("Cloned repo successfully.")