from abc import abstractmethod
from typing import Generator, Tuple
from pathlib import Path
import csv
from tqdm import tqdm

from nboost.helpers import count_lines
from nboost.logger import set_logger
from nboost.indexers import defaults
from nboost import PKG_PATH


CSV_PATH_DEFAULT = defaults.file
INDEX_NAME_DEFAULT = defaults.index_name
ID_COL_DEFAULT = defaults.id_col
HOST_DEFAULT = defaults.host
PORT_DEFAULT = defaults.port
DELIM_DEFAULT = defaults.delim
SHARDS_DEFAULT = defaults.shards
VERBOSE_DEFAULT = defaults.verbose


class CSVIterator:
    """An iterable generator that reads a csv file."""

    def __init__(self, file_path: Path, delim: str = DELIM_DEFAULT):
        self.file_path = file_path
        self.delim = delim

    def __iter__(self) -> Generator[Tuple[str, str], None, None]:
        with self.file_path.open() as file:
            for cid, passage in csv.reader(file, delimiter=self.delim):
                yield cid, passage


class CSVFinder:
    """A utility class that attempts to find a csv file."""

    def __init__(self, file_path: Path = CSV_PATH_DEFAULT):
        self.file_path = file_path

    def find(self) -> Path:
        if self.file_path.exists():
            return self.file_path
        pkg_path = PKG_PATH.joinpath('resources').joinpath(self.file_path)
        if pkg_path.exists():
            return pkg_path
        raise FileNotFoundError("Could not find CSV file.")


class BaseIndexer:
    """An object that sends a csv to a given search api."""

    def __init__(
        self,
        file_path: Path = CSV_PATH_DEFAULT,
        index_name: str = INDEX_NAME_DEFAULT,
        id_col: int = ID_COL_DEFAULT,
        host: str = HOST_DEFAULT,
        port: int = PORT_DEFAULT,
        delim: str = DELIM_DEFAULT,
        shards: int = SHARDS_DEFAULT,
        verbose: bool = VERBOSE_DEFAULT,
    ):
        self.file_path = file_path
        self.index_name = index_name
        self.id_col = id_col
        self.host = host
        self.port = port
        self.delim = delim
        self.shards = shards
        self.logger = set_logger(self.__class__.__name__, verbose=verbose)

    def index(self):
        """Uses the csv_generator() to send the csv to the index"""
        csv_finder = CSVFinder(self.file_path)
        csv_path = csv_finder.find()
        csv_iter = CSVIterator(csv_path, self.delim)
        num_lines = count_lines(csv_path)
        with tqdm(total=num_lines, desc=csv_path.name) as pbar:
            for cid, passage in csv_iter:
                self.send_to_index(cid, passage)
                pbar.update()

    @abstractmethod
    def send_to_index(self, cid: str, passage: str):
        """Sends the given passage to the search api."""


class ElasticsearchIndexer(BaseIndexer):
    """An indexer for Elasticsearch."""

    @abstractmethod
    def send_to_index(self, cid: str, passage: str):
        """
        Indexes the given passage in Elasticsearch.

        :param cid: the id of the passage
        :param passage: the text of the passage
        """


class SolrIndexer(BaseIndexer):
    """An indexer for Solr."""

    @abstractmethod
    def send_to_index(self, cid: str, passage: str):
        """
        Indexes the given passage in Solr.

        :param cid: the id of the passage
        :param passage: the text of the passage
        """