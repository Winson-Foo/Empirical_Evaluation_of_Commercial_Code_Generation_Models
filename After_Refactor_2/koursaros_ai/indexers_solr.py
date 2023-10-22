from typing import Dict

import pysolr

from nboost.indexers.base import BaseIndexer

class SolrIndexer(BaseIndexer):
    """
    Index csvs in Solr
    """

    CORE_NAME = "travel"
    PASSAGE_FIELD = "passage_t"
    ID_FIELD = "id"

    def __init__(self, num_shards: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.num_shards = num_shards
        self.solr = pysolr.Solr(self.get_core_url(), timeout=10000)

    def get_core_url(self) -> str:
        return f"http://{self.host}:{self.port}/solr/{self.CORE_NAME}/"

    def format_passage(self, passage: str, cid: str) -> Dict[str, str]:
        """
        Format a passage for indexing
        """
        body = {
            self.PASSAGE_FIELD: passage
        }
        if cid is not None:
            body[self.ID_FIELD] = cid
        return body

    def get_formatted_passages(self) -> List[Dict[str, str]]:
        """
        Get a list of formatted passages for indexing
        """
        formatted_passages = [self.format_passage(passage, cid=cid) for cid, passage in self.csv_generator()]
        return formatted_passages

    def index_passages(self, formatted_passages: List[Dict[str, str]]) -> None:
        """
        Index the given list of formatted passages to Solr
        """
        self.logger.info("Indexing %s...", self.file)
        self.solr.add(formatted_passages)
        self.solr.optimize()

    def index(self) -> None:
        """
        Send csv to Solr index
        """
        self.logger.info("Setting up Solr index...")
        self.solr_admin = pysolr.SolrCoreAdmin(self.get_core_url())
        self.solr_admin.create(self.CORE_NAME, self.num_shards, False)

        formatted_passages = self.get_formatted_passages()
        self.index_passages(formatted_passages)