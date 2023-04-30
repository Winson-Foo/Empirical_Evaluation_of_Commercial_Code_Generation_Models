from typing import Dict
from nboost.indexers.base import BaseIndexer
from pysolr import Solr, SolrCoreAdmin

class SolrIndexer(BaseIndexer):
    """
    Indexes csv data in Solr
    """

    solr_url = "http://localhost:8983/solr/travel/"

    def __init__(self, shards: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.solr = Solr(SolrIndexer.solr_url, timeout=10000)

    def format_document(self, passage: str, cid: str) -> Dict[str, str]:
        """
        Formats a document for indexing in Solr
        """
        document = {'passage': passage}
        if cid is not None:
            document['id'] = cid
        return document

    def index_data(self):
        """
        Indexes csv data in Solr
        """
        try:
            self.logger.info('Setting up Solr index...')
            self.solr.core_admin.reload()
            self.logger.info('Indexing %s...' % self.file)
            documents = [self.format_document(passage, cid=cid) for cid, passage in self.csv_generator()]
            self.solr.add(documents)
            self.solr.commit()
            self.logger.info('Indexing completed')
        except Exception as e:
            self.logger.error(f'Error indexing csv data: {str(e)}')