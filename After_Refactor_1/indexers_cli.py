"""NBoost command line interface"""
from argparse import ArgumentParser
from typing import List, Type

import termcolor
from nboost.indexers.base import BaseIndexer
from nboost.helpers import import_class
from nboost.indexers import defaults
from nboost.maps import INDEXER_MAP


TAG = termcolor.colored('NBoost Indexer', 'cyan', attrs=['underline'])
DESCRIPTION = ('This is the {}. This command line utility can be used to send '
               'a csv to a search api for indexing.').format(TAG)

# Command line argument constants
FILE = 'file'
INDEX_NAME = 'index_name'
ID_COL = 'id_col'
HOST = 'host'
PORT = 'port'
DELIM = 'delim'
SHARDS = 'shards'
INDEXER = 'indexer'
VERBOSE = 'verbose'


def set_parser() -> ArgumentParser:
    """Create argument parser with default nboost-index cli arguments."""
    parser = ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--verbose', type=bool, default=defaults.verbose, help='Turn on detailed logging.')
    parser.add_argument('--file', type=str, default=defaults.file, help='Path of the csv to send to the index.')
    parser.add_argument('--index_name', type=str, default=defaults.index_name, help='Name of the index to send to.')
    parser.add_argument('--host', type=str, default=defaults.host, help='Host of the search api server.')
    parser.add_argument('--port', type=int, default=defaults.port, help='Port of the server.')
    parser.add_argument('--delim', type=str, default=defaults.delim, help='Csv delimiter.')
    parser.add_argument('--shards', type=int, default=defaults.shards, help='Number of index shards to create.')
    parser.add_argument('--id_col', action='store_true', help='Whether to index each doc with an id (should be first col in csv).')
    parser.add_argument('--indexer', type=str, default=defaults.indexer, help='The indexer class.')
    return parser


def get_indexer(indexer_class: str) -> Type[BaseIndexer]:
    """Get indexer instance from indexer class."""
    indexer_module = INDEXER_MAP[indexer_class]
    indexer = import_class(indexer_module, indexer_class)
    return indexer


def index_csv(args: dict) -> None:
    """Index a given csv using NBoost."""
    indexer_class = args.pop(INDEXER)
    indexer = get_indexer(indexer_class)
    indexer(**args).index()


def main(argv: List[str] = None) -> None:
    """Execute the command line interface."""
    parser = set_parser()
    args = vars(parser.parse_args(argv))
    index_csv(args)


if __name__ == "__main__":
    main()