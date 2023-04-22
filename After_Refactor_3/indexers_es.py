from typing import Dict, Generator, Tuple
from elasticsearch.exceptions import RequestError
from elasticsearch.helpers import bulk
from elasticsearch import Elasticsearch
from nboost.indexers.base import BaseIndexer


class ESIndexer(BaseIndexer):
    """Index passages in Elasticsearch"""

    INDEX_SETTINGS = {'settings': {'index': {'number_of_shards': 1}}}
    INDEX_TYPE = '_doc'
    PASSAGE_FIELD = 'passage'

    def __init__(self, shards: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.index_settings = {'settings': {'index': {'number_of_shards': shards}}}

    def format_passage_for_indexing(self, passage: str, passage_id: str) -> Dict:
        """Format a passage for indexing in Elasticsearch"""
        body = {
            '_index': self.index_name,
            '_type': ESIndexer.INDEX_TYPE,
            '_source': {ESIndexer.PASSAGE_FIELD: passage}
        }

        if passage_id is not None:
            body['_id'] = passage_id

        return body

    def index_passages(self):
        """Send passages to Elasticsearch index"""
        self.logger.info('Setting up Elasticsearch index...')
        elastic = Elasticsearch(host=self.host, port=self.port, timeout=10000)
        try:
            self.logger.info('Creating index %s...' % self.index_name)
            elastic.indices.create(self.index_name, self.index_settings)
        except RequestError:
            self.logger.info('Index already exists, skipping...')

        self.logger.info('Indexing %s...' % self.file)
        actions_generator = self.get_actions_generator()
        bulk(elastic, actions=actions_generator)

    def get_actions_generator(self) -> Generator[Tuple, None, None]:
        """Return a generator of formatted passages for bulk indexing in Elasticsearch"""
        passages = self.csv_generator()
        formatted_passages = (self.format_passage_for_indexing(passage=passage, passage_id=passage_id) for passage_id, passage in passages)
        return formatted_passages