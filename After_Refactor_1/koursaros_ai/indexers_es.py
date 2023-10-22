from typing import Dict
from elasticsearch.exceptions import RequestError
from elasticsearch.helpers import bulk
from elasticsearch import Elasticsearch
from nboost.indexers.base import BaseIndexer

class ESIndexer(BaseIndexer):
    """Index documents in Elasticsearch"""

    ES_INDEX_TYPE = '_doc'

    def __init__(self, shards: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.shards = shards
        self.mapping = {'settings': {'index': {'number_of_shards': self.shards}}}

    def format_document(self, document: Dict) -> Dict:
        """Format a document for indexing"""
        document_body = {'_index': self.index_name, '_type': ESIndexer.ES_INDEX_TYPE, '_source': document['passage']}
        if document.get('document_id'):
            document_body['_id'] = document['document_id']
        return document_body

    def create_es_index(self, elastic: Elasticsearch):
        """Create the Elasticsearch index"""
        try:
            self.logger.info('Creating index %s...' % self.index_name)
            elastic.indices.create(self.index_name, self.mapping)
        except RequestError:
            self.logger.info('Index already exists, skipping...')

    def get_es_client(self) -> Elasticsearch:
        """Get the Elasticsearch client"""
        return Elasticsearch(host=self.host, port=self.port, timeout=10000)

    def index_documents(self):
        """Send documents to Elasticsearch"""
        self.logger.info('Setting up Elasticsearch index...')
        es_client = self.get_es_client()
        self.create_es_index(es_client)

        self.logger.info('Indexing %s...' % self.file)

        documents = self.csv_generator()
        actions = [self.format_document(doc) for doc in documents]

        try:
            bulk(es_client, actions=actions)
        except Exception as err:
            self.logger.error('Failed to bulk index documents: %s' % str(err))