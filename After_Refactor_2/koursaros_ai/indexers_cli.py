from argparse import ArgumentParser
from typing import List, Type
import termcolor
from nboost.indexers.base import BaseIndexer
from nboost.helpers import import_class
from nboost.indexers import defaults
from nboost.maps import INDEXER_MAP


TAG = termcolor.colored('NBoost Indexer', 'cyan', attrs=['underline'])
DESCRIPTION = ('This is the %s. This command line utility can be used to send '
               'a csv to a search api for indexing.' % TAG)

# Constants for command line argument help messages
FILE_HELP = 'Path of the CSV file to send to the index.'
INDEX_NAME_HELP = 'Name of the index to send to.'
ID_COL_HELP = 'Whether to index each document with an ID (should be the first column in the CSV file).'
HOST_HELP = 'Host of the search API server.'
PORT_HELP = 'Port of the server.'
DELIM_HELP = 'Delimiter for the CSV file.'
SHARDS_HELP = 'Number of index shards to create.'
INDEXER_HELP = 'The indexer class to use.'
VERBOSE_HELP = 'Turn on detailed logging.'


def set_parser() -> ArgumentParser:
    """Add default nboost-index command line arguments to a given parser."""
    parser = ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--verbose', type=bool, default=defaults.verbose, help=VERBOSE_HELP)
    parser.add_argument('--file', type=str, default=defaults.file, help=FILE_HELP)
    parser.add_argument('--index_name', type=str, default=defaults.index_name, help=INDEX_NAME_HELP)
    parser.add_argument('--host', type=str, default=defaults.host, help=HOST_HELP)
    parser.add_argument('--port', type=int, default=defaults.port, help=PORT_HELP)
    parser.add_argument('--delim', type=str, default=defaults.delim, help=DELIM_HELP)
    parser.add_argument('--shards', type=int, default=defaults.shards, help=SHARDS_HELP)
    parser.add_argument('--id_col', action='store_true', help=ID_COL_HELP)
    parser.add_argument('--indexer', type=str, default=defaults.indexer, help=INDEXER_HELP)
    return parser


def get_indexer_class(indexer_name: str) -> Type[BaseIndexer]:
    """Return the BaseIndexer subclass corresponding to the given indexer name."""
    indexer_module = INDEXER_MAP[indexer_name]
    return import_class(indexer_module, indexer_name)


def perform_indexing(indexer_class: Type[BaseIndexer], params: dict):
    """Create and run an instance of the given indexer class with the provided parameters."""
    indexer = indexer_class(**params)
    indexer.index()


def parse_args(argv: List[str]) -> dict:
    """Parse the given command line arguments and return them as a dictionary."""
    parser = set_parser()
    return vars(parser.parse_args(argv))


def run_nboost_indexer(argv: List[str] = None):
    """Run the NBoost Indexer command line utility with the given arguments."""
    params = parse_args(argv)
    indexer_class = get_indexer_class(params.pop('indexer'))
    perform_indexing(indexer_class, params)


if __name__ == "__main__":
    run_nboost_indexer()