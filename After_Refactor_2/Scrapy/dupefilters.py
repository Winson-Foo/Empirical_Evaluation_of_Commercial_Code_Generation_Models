import logging
from pathlib import Path
from typing import Optional, Set, Type, TypeVar

from scrapy.http.request import Request
from scrapy.settings import BaseSettings
from scrapy.spiders import Spider
from scrapy.utils.deprecate import ScrapyDeprecationWarning
from scrapy.utils.job import job_dir
from scrapy.utils.request import RequestFingerprinter, referer_str
from twisted.internet.defer import Deferred

BaseDupeFilterTV = TypeVar("BaseDupeFilterTV", bound="BaseDupeFilter")
logger = logging.getLogger(__name__)

class BaseDupeFilter:
    @classmethod
    def from_settings(cls: Type[BaseDupeFilterTV], settings: BaseSettings) -> BaseDupeFilterTV:
        return cls()

    def request_seen(self, request: Request) -> bool:
        return False

    def open(self) -> Optional[Deferred]:
        pass

    def close(self, reason: str) -> Optional[Deferred]:
        pass

class RFPDupeFilter(BaseDupeFilter):
    def __init__(self, path: Optional[str] = None, debug: bool = False, *, fingerprinter=None) -> None:
        self.file = None
        self.fingerprinter = fingerprinter or RequestFingerprinter()
        self.fingerprints: Set[str] = set()
        self.logdupes = True
        self.debug = debug
        if path:
            with Path(path, "requests.seen").open("a+", encoding="utf-8") as f:
                self.file = f
                self.fingerprints.update(x.rstrip() for x in self.file)

    @classmethod
    def from_settings(cls: Type['RFPDupeFilter'], settings: BaseSettings, *, fingerprinter=None) -> 'RFPDupeFilter':
        debug = settings.getbool("DUPEFILTER_DEBUG")
        try:
            return cls(job_dir(settings), debug, fingerprinter=fingerprinter)
        except TypeError:
            logging.warning(
                "RFPDupeFilter subclasses must either modify their '__init__' "
                "method to support a 'fingerprinter' parameter or reimplement "
                "the 'from_settings' class method.", ScrapyDeprecationWarning)
            result = cls(job_dir(settings), debug)
            result.fingerprinter = fingerprinter
            return result

    @classmethod
    def from_crawler(cls: Type['RFPDupeFilter'], crawler) -> 'RFPDupeFilter':
        try:
            return cls.from_settings(crawler.settings, fingerprinter=crawler.request_fingerprinter)
        except TypeError:
            logging.warning(
                "RFPDupeFilter subclasses must either modify their overridden "
                "'__init__' method and 'from_settings' class method to "
                "support a 'fingerprinter' parameter, or reimplement the "
                "'from_crawler' class method.", ScrapyDeprecationWarning)
            result = cls.from_settings(crawler.settings)
            result.fingerprinter = crawler.request_fingerprinter
            return result

    def request_seen(self, request: Request) -> bool:
        fp = self.request_fingerprint(request)
        if fp in self.fingerprints:
            return True
        self.fingerprints.add(fp)
        if self.file:
            self.file.write(fp + "\n")
        return False

    def request_fingerprint(self, request: Request) -> str:
        return self.fingerprinter.fingerprint(request).hex()

    def close(self, reason: str) -> None:
        if self.file:
            self.file.close()

    def log(self, request: Request, spider: Spider) -> None:
        if self.debug:
            msg = "Filtered duplicate request: %(request)s (referer: %(referer)s)"
            args = {"request": request, "referer": referer_str(request)}
            logger.debug(msg, args, extra={"spider": spider})
        elif self.logdupes:
            msg = (
                "Filtered duplicate request: %(request)s"
                " - no more duplicates will be shown"
                " (see DUPEFILTER_DEBUG to show all duplicates)"
            )
            logger.debug(msg, {"request": request}, extra={"spider": spider})
            self.logdupes = False

        spider.crawler.stats.inc_value("dupefilter/filtered", spider=spider) 