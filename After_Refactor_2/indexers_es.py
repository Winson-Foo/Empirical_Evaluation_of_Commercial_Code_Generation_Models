from typing import Dict
from elasticsearch.exceptions import RequestError
from elasticsearch.helpers import bulk
from elasticsearch import Elasticsearch
from nboost.indexers.base import BaseIndexer


class ESIndexer(BaseIndexer):
    """Index csvs in Elasticsearch"""

    SETTINGS = {'settings': {'index': {'number_of_shards': 1}}}
    DOC_TYPE = '_doc'

    def __init__(self, shards: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.mapping = ESIndexer.SETTINGS

    def format_to_index(self, passage: str, cid: str) -> Dict:
        """Format a passage for indexing"""
        source = {"passage": passage}
        body = {
            '_index': self.index_name,
            '_type': ESIndexer.DOC_TYPE,
            '_source': source
        }

        if cid is not None:
            body['_id'] = cid

        return body

    def _create_index(self, elastic):
        if elastic.indices.exists(self.index_name):
            self.logger.info('Index already exists, skipping...')
        else:
            self.logger.info(f'Creating index {self.index_name}...')
            elastic.indices.create(self.index_name, self.mapping)

    def _index_csv(self, elastic):
        actions = (self.format_to_index(passage, cid=cid) for cid, passage in self.csv_generator())
        bulk(elastic, actions=actions)

    def _connect_to_es(self):
        return Elasticsearch(host=self.host, port=self.port, timeout=10000)

    def index_csv_to_es(self):
        """send csv to ES index"""
        self.logger.info('Setting up Elasticsearch index...')

        elastic = self._connect_to_es()
        self._create_index(elastic)
        self.logger.info(f'Indexing {self.file}...')

        self._index_csv(elastic)