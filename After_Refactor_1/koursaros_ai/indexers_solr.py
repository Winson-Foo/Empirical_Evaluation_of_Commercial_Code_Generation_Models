from typing import Dict, Optional
import logging

from nboost.indexers.base import BaseIndexer
from pysolr import Solr, SolrCoreAdmin

LOG_FORMAT = "%(levelname)s %(asctime)s %(module)s:%(lineno)d %(message)s"
SOLR_URL = "http://localhost:8983/solr"
SOLR_CORE = "travel"

class SolrIndexer(BaseIndexer):
    """Index passages in Solr"""
    def __init__(self, shards: int = 1, solr_url: str = SOLR_URL,
                 solr_core: str = SOLR_CORE, **kwargs) -> None:
        super().__init__(**kwargs)
        self.solr_url = solr_url
        self.solr_core = solr_core
        self.solr = Solr(f"{solr_url}/{solr_core}/", timeout=10000)
        self.logger = logging.getLogger(__name__)

    def format(self, passage: str, cid: Optional[str] = None) -> Dict[str, str]:
        """Format a passage for indexing"""
        body = {
            "passage_t": passage
        }

        if cid is not None:
            body["id"] = cid

        return body

    def index(self) -> None:
        """Index passages in Solr"""
        self.logger.info("Setting up Solr index...")
        self.logger.debug(f"Solr URL: {self.solr_url}")
        self.logger.debug(f"Solr core: {self.solr_core}")

        self.logger.info(f"Indexing {self.file}...")
        passages = [self.format(passage, cid=cid) for cid, passage in self.csv_generator()]
        self.logger.debug(f"Passages to index: {passages}")
        
        self.solr.add(passages)
        self.logger.info("Committing changes to Solr...")
        self.solr.commit()
        self.logger.debug("Optimizing Solr index...")
        self.solr.optimize()
        self.logger.info("Done.")