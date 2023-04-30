from abc import abstractmethod
from typing import Generator
from pathlib import Path
import csv
from tqdm import tqdm

import nboost
from nboost.helpers import count_lines
from nboost.logger import set_logger
from nboost.indexers import defaults


class BaseIndexer:
    """An object that sends a csv to a given search api."""

    def __init__(self, **config):
        """
        :param file: name of the file to read from
        :param index_name: name of the index
        :param id_col: column number of the id
        :param host: host of the search api server
        :param port: port the the server
        :param delim: delimiter for the csv file
        :param shards: number of shards for the index
        """
        self.file = config.get('file', defaults.file)
        self.index_name = config.get('index_name', defaults.index_name)
        self.id_col = config.get('id_col', defaults.id_col)
        self.host = config.get('host', defaults.host)
        self.port = config.get('port', defaults.port)
        self.delim = config.get('delim', defaults.delim)
        self.shards = config.get('shards', defaults.shards)
        self.logger = set_logger(self.__class__.__name__, verbose=config.get('verbose', defaults.verbose))

    def csv_reader(self, path: Path) -> Generator:
        self.logger.info('Estimating completion size...')
        num_lines = count_lines(path)
        with path.open() as file:
            with tqdm(total=num_lines, desc=path.name) as pbar:
                for cid, passage in csv.reader(file, delimiter=self.delim):
                    yield cid, passage
                    pbar.update()

    def csv_generator(self) -> Generator:
        """Check for the csv in the current working directory first, then
        search for it in the package.

        Generates id_col, passage
        """
        cwd_path = Path().joinpath(self.file).absolute()
        pkg_path = nboost.PKG_PATH.joinpath('resources').joinpath(self.file)

        if cwd_path.exists():
            path = cwd_path
        elif pkg_path.exists():
            path = pkg_path
        else:
            error_message = f'Could not find {pkg_path} or {cwd_path}'
            self.logger.error(error_message)
            raise FileNotFoundError(error_message)

        return self.csv_reader(path)

    @abstractmethod
    def index(self):
        """uses the csv_generator() to send the csv to the index"""
        pass