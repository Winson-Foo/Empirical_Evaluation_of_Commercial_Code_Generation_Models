import os
import shutil
from git import Repo
import logging
from .common import sha256_hash, home_directory_path

FOLDER = ".foolbox_zoo"


def clone(git_uri: str, overwrite: bool = False) -> str:
    """Clones a remote git repository to a local path.

    Args:
        git_uri: The URI to the git repository to be cloned.
        overwrite: Whether or not to overwrite the local path.

    Returns:
        The generated local path where the repository has been cloned to.
    """
    local_path = generate_local_path(git_uri)
    
    if os.path.exists(local_path) and overwrite:
        remove_local_path(local_path)
    
    if not os.path.exists(local_path):
        clone_repo(git_uri, local_path)
    else:
        log_info("Git repository already exists locally.")

    return local_path


def generate_local_path(git_uri: str) -> str:
    """Generates the local path for cloning the repository.

    Args:
        git_uri: The URI to the git repository.

    Returns:
        The generated local path.
    """
    hash_digest = sha256_hash(git_uri)
    return home_directory_path(FOLDER, hash_digest)


def remove_local_path(local_path: str) -> None:
    """Removes the local path if it exists.

    Args:
        local_path: The local path to be removed.
    """
    log_info(f"Removing local path: {local_path}")
    shutil.rmtree(local_path, ignore_errors=True)


def clone_repo(git_uri: str, local_path: str) -> None:
    """Clones the git repository to the local path.

    Args:
        git_uri: The URI to the git repository.
        local_path: The local path for cloning the repository.
    """
    log_info(f"Cloning repo {git_uri} to {local_path}")
    try:
        Repo.clone_from(git_uri, local_path)
        log_info("Cloned repo successfully.")
    except Exception as e:
        logging.exception("Failed to clone repository", exc_info=e)
        raise GitCloneError("Failed to clone repository")


def log_info(message: str) -> None:
    """Logs an info message.

    Args:
        message: The message to be logged.
    """
    logging.info(message)