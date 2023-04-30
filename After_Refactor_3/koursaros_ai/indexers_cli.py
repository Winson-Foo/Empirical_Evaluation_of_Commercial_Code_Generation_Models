from argparse import ArgumentParser
from typing import List, Type
import termcolor

from nboost.indexers.base import BaseIndexer
from nboost.helpers import import_class
from nboost.indexers import defaults
from nboost.maps import INDEXER_MAP


# CLI argument descriptions
VERBOSE_HELP = 'turn on detailed logging'
FILE_HELP = 'path of the csv to send to the index'
INDEX_NAME_HELP = 'name of the index send to'
HOST_HELP = 'host of the search api server'
PORT_HELP = 'port of the server'
DELIM_HELP = 'csv delimiter'
SHARDS_HELP = 'number of index shards to create'
ID_COL_HELP = 'whether to index each doc with an id (should be first col in csv)'
INDEXER_HELP = 'the indexer class'

# CLI argument defaults
VERBOSE_DEFAULT = defaults.verbose
FILE_DEFAULT = defaults.file
INDEX_NAME_DEFAULT = defaults.index_name
HOST_DEFAULT = defaults.host
PORT_DEFAULT = defaults.port
DELIM_DEFAULT = defaults.delim
SHARDS_DEFAULT = defaults.shards
INDEXER_DEFAULT = defaults.indexer

# Output formatting
TAG = termcolor.colored('NBoost Indexer', 'cyan', attrs=['underline'])
DESCRIPTION = f'This is the {TAG}. This command line utility can be used to send a csv to a search api for indexing.'


def set_parser() -> ArgumentParser:
    parser = ArgumentParser(description=DESCRIPTION)

    parser.add_argument('--verbose', type=type(VERBOSE_DEFAULT), default=VERBOSE_DEFAULT, help=VERBOSE_HELP)
    parser.add_argument('--file', type=type(FILE_DEFAULT), default=FILE_DEFAULT, help=FILE_HELP)
    parser.add_argument('--index_name', type=type(INDEX_NAME_DEFAULT), default=INDEX_NAME_DEFAULT, help=INDEX_NAME_HELP)
    parser.add_argument('--host', type=type(HOST_DEFAULT), default=HOST_DEFAULT, help=HOST_HELP)
    parser.add_argument('--port', type=type(PORT_DEFAULT), default=PORT_DEFAULT, help=PORT_HELP)
    parser.add_argument('--delim', type=type(DELIM_DEFAULT), default=DELIM_DEFAULT, help=DELIM_HELP)
    parser.add_argument('--shards', type=type(SHARDS_DEFAULT), default=SHARDS_DEFAULT, help=SHARDS_HELP)
    parser.add_argument('--id_col', action='store_true', help=ID_COL_HELP)
    parser.add_argument('--indexer', type=type(INDEXER_DEFAULT), default=INDEXER_DEFAULT, help=INDEXER_HELP)

    return parser


def create_indexer(args: dict) -> BaseIndexer:
    indexer_class = args.pop('indexer')
    indexer_module = INDEXER_MAP[indexer_class]
    indexer = import_class(indexer_module, indexer_class)  # type: Type[BaseIndexer]
    return indexer(**args)


def run_indexer(indexer: BaseIndexer) -> None:
    indexer.index()


def main(argv: List[str] = None) -> None:
    parser = set_parser()
    args = vars(parser.parse_args(argv))

    indexer = create_indexer(args)
    run_indexer(indexer)


if __name__ == '__main__':
    main()