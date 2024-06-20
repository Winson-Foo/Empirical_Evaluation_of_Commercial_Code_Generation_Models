import os
import shutil
from git import Repo
from hashlib import sha256
from typing import Tuple
from .common import home_directory_path

FOLDER = ".foolbox_zoo"


class GitCloneError(RuntimeError):
    pass


def clone(git_uri: str, overwrite: bool = False) -> str:
    """Clones a remote git repository to a local path.

    Args:
        git_uri: The URI to the git repository to be cloned.
        overwrite: Whether or not to overwrite the local path.

    Returns:
        The generated local path where the repository has been cloned to.
        
    Raises:
        GitCloneError: If the repository could not be cloned.
    """
    hash_digest = get_git_uri_hash(git_uri)
    local_path = get_local_path(hash_digest)

    if os.path.exists(local_path):
        if overwrite:
            remove_local_path(local_path)
        else:
            logging.info("Git repository already exists locally.")
            return local_path

    try:
        clone_repo(git_uri, local_path)
    except Exception as e:
        raise GitCloneError("Failed to clone repository", e)

    return local_path


def get_git_uri_hash(git_uri: str) -> str:
    """Generates a hash digest for the given git URI.

    Args:
        git_uri: The URI to the git repository.

    Returns:
        The hash digest of the git URI.
    """
    hash_object = sha256(git_uri.encode())
    return hash_object.hexdigest()


def get_local_path(hash_digest: str) -> str:
    """Generates the local path where the repository will be cloned to.

    Args:
        hash_digest: The hash digest of the git URI.

    Returns:
        The local path where the repository will be cloned to.
    """
    return home_directory_path(FOLDER, hash_digest)


def remove_local_path(local_path: str) -> None:
    """Removes the local path.

    Args:
        local_path: The local path to be removed.
    """
    shutil.rmtree(local_path, ignore_errors=True)


def clone_repo(git_uri: str, local_path: str) -> None:
    """Clones the git repository to the specified local path.

    Args:
        git_uri: The URI of the git repository to be cloned.
        local_path: The local path where the repository will be cloned to.

    Raises:
        GitCloneError: If the repository could not be cloned.
    """
    try:
        logging.info("Cloning repository %s to %s", git_uri, local_path)
        Repo.clone_from(git_uri, local_path)
        logging.info("Cloned repository successfully.")
    except Exception as e:
        logging.exception("Failed to clone repository", exc_info=e)
        raise GitCloneError("Failed to clone repository")