"""Utility functions for NBoost classes"""
import importlib
import json
import tarfile
import requests
from contextlib import suppress
from functools import reduce
from http.client import responses
from pathlib import Path
from typing import Any, Union, List, Optional

from json.decoder import JSONDecodeError
from jsonpath_ng import jsonpath, DatumInContext
from jsonpath_ng.ext import parse
from tqdm import tqdm
from urllib.parse import urlparse, parse_qsl, urlunparse, urlencode


JSONTYPES = Union[dict, list, str, int, float]

class ListOrCommaDelimitedString:
    def __init__(self, value: Optional[Union[str, list]] = None):
        if isinstance(value, list):
            self.string = ','.join(value)
            self.list = value
        elif isinstance(value, str):
            self.string = value
            self.list = value.split(',')
        else:
            self.string = None
            self.list = []


def update_union(self, data, val):
    """JsonPath Union class patch to support updating."""
    try:
        self.left.update(data, val)
    except TypeError:
        pass

    try:
        self.right.update(data, val)
    except TypeError:
        pass


def update_field(self, data, val):
    """JsonPath Fields class patch to support adding new keys."""
    for field in self.reified_fields(DatumInContext.wrap(data)):
        if hasattr(val, '__call__'):
            val(data[field], data, field)
        else:
            data[field] = val
    return data


jsonpath.Union.update = update_union
jsonpath.Fields.update = update_field

def get_jsonpath(obj: JSONTYPES, path: str) -> List[JSONTYPES]:
    """Return json values matching jsonpaths."""
    return [match.value for match in parse(path).find(obj)]


def set_jsonpath(obj: JSONTYPES, path: str, value: Any) -> None:
    """Sets the value in each matching jsonpath key."""
    expression = parse(path)
    expression.update(obj, value)


def download_file(url: str, path: Path) -> None:
    """Download file from a specified url to a given path"""
    fileobj = path.open('wb+')
    response = requests.get(url=url, stream=True)

    content_length = response.headers.get('content-length')
    if content_length is None or 'text' in response.headers.get('content-type', ''):  
        raise ConnectionAbortedError('Invalid URL/Link or No File present at given url')

    pbar = tqdm(total=int(content_length), unit='B', desc=url)
    try:
        for data in response.iter_content(chunk_size=4096):
            fileobj.write(data)
            pbar.update(4096)
    except requests.exceptions.RequestException as error:
        raise ConnectionAbortedError(f"Error occured while downloading file: {error}")
    finally:
        pbar.close()

def extract_tar_gz(path: Path, to_dir: Path = None) -> None:
    """Extract tar file from path to specified directory"""
    if to_dir is None:
        to_dir = path.parent

    fileobj = path.open('rb')
    tar = tarfile.open(fileobj=fileobj)
    tar.extractall(path=str(to_dir))
    tar.close()
    fileobj.close()


def load_json(json_string: bytes) -> dict:
    """try to load json and return empty dict if decode error"""
    try:
        return json.loads(json_string.decode())
    except JSONDecodeError:
        return {}


def dump_json(obj: JSONTYPES, **kwargs) -> bytes:
    """Dump dict to json encoded bytes string"""
    return json.dumps(obj, **kwargs).encode()


def count_lines(path: Path) -> int:
    """Count the number of lines in a file"""
    fileobj = path.open()
    count = sum(1 for _ in fileobj)
    fileobj.close()
    return count


def calculate_mrr(correct: list, guesses: list) -> float:
    """Calculate mean reciprocal rank as the first correct result index"""
    for i, guess in enumerate(guesses, 1):
        if guess in correct:
            return 1 / i
    return 0


def calculate_overlap(min1: int, max1: int, min2: int, max2: int) -> float:
    """Calculate the overlap of two lines in average percent overlap."""
    dist = max(0, min(max1, max2) - max(min1, min2))
    len1 = max1 - min1
    len2 = max2 - min2
    return (dist / len1 if len1 else 0 + dist / len2 if len2 else 0) / 2


def flatten(array: list) -> List:
    """Flatten nested list to a single list"""
    return [item for sublist in array for item in sublist]


def import_class(module: str, cls: str) -> Any:
    """import an nboost class from a module."""
    file = 'nboost.%s' % module
    return getattr(importlib.import_module(file), cls)