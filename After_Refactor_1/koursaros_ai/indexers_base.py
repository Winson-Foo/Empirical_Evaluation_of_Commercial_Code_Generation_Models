import os
import csv
from abc import abstractmethod
from typing import Generator
from tqdm import tqdm
from fastapi import FastAPI
from nboost.indexers import defaults
from nboost import PKG_PATH


class BaseIndexer:
    """An object that sends a csv to a given search api."""

    def __init__(self, file_path: str = defaults.file,
                 index_name: str = defaults.index_name,
                 index_id: str = defaults.id_col,
                 host: str = defaults.host,
                 port: int = defaults.port,
                 delimiter: str = defaults.delim,
                 shards: int = defaults.shards, verbose: bool = defaults.verbose,
                 **kwargs):
        """
        :param file_path: Path to csv file
        :param index_name: Name of the index
        :param index_id: Name of the ID column
        :param host: Host of the search api server
        :param port: Port the server
        :param delimiter: Delimiter used in csv file
        :param shards: Number of shards for the index
        :param verbose: Logging level
        """
        self.file_path = file_path
        self.index_name = index_name
        self.index_id = index_id
        self.host = host
        self.port = port
        self.delimiter = delimiter
        self.shards = shards
        self.app = FastAPI()
        self.logger = self.app.logger
        self.logger.setLevel(verbose)

    def create_file_path(self) -> str:
        """ Generate file path """
        cwd_file = os.path.join(os.getcwd(), self.file_path)
        pkg_file = os.path.join(PKG_PATH, 'resources', self.file_path)
        return cwd_file if os.path.exists(cwd_file) else pkg_file

    def generate_documents(self) -> Generator:
        """ Generate documents from csv file """
        file_path = self.create_file_path()
        num_lines = sum(1 for _ in open(file_path))
        with open(file_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=self.delimiter)
            for row in tqdm(csv_reader, desc=self.file_path, total=num_lines):
                yield row[self.index_id], row[1]

    @abstractmethod
    def index(self):
        """uses the csv_generator() to send the csv to the index"""
        pass


class CustomIndexer(BaseIndexer):
    def index(self):
        for cid, passage in self.generate_documents():
            # your indexing code here
            self.logger.info(f"Indexing document {cid}")