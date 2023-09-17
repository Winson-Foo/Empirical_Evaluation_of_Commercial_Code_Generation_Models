import hashlib
from pathlib import Path

def sha256_hash(string: str) -> str:
    sha256 = hashlib.sha256()
    sha256.update(string.encode('utf-8'))
    return sha256.hexdigest()

def home_directory_path(*paths) -> str:
    return str(Path.home().joinpath(*paths))