import logging
from pathlib import Path
from typing import Optional, Set, Type, TypeVar
from warnings import warn

from twisted.internet.defer import Deferred

from scrapy.http.request import Request
from scrapy.settings import BaseSettings
from scrapy.spiders import Spider
from scrapy.utils.deprecate import ScrapyDeprecationWarning
from scrapy.utils.job import job_dir
from scrapy.utils.request import RequestFingerprinter, referer_str

TDupeFilter = TypeVar('TDupeFilter', bound='DupeFilter')


class DupeFilter:
    def __init__(self, path: Optional[str] = None, debug: bool = False,
                 fingerprinter=None) -> None:
        self.file = None
        self.fingerprinter = fingerprinter or RequestFingerprinter()
        self.fingerprints: Set[str] = set()
        self.log_duplicates = True
        self.debug = debug
        self.logger = logging.getLogger(__name__)
        if path:
            self.file = Path(path, "requests.seen").open("a+", encoding="utf-8")
            self.file.seek(0)
            self.fingerprints.update(x.rstrip() for x in self.file)

    @classmethod
    def from_settings(cls: Type[TDupeFilter], settings: BaseSettings,
                      *, fingerprinter=None) -> TDupeFilter:
        debug = settings.getbool('DUPEFILTER_DEBUG')
        try:
            return cls(job_dir(settings), debug, fingerprinter=fingerprinter)
        except TypeError:
            warn("DupeFilter subclasses must either modify their '__init__' "
                 "method to support a 'fingerprinter' parameter or reimplement "
                 "the 'from_settings' class method.", ScrapyDeprecationWarning)
            result = cls(job_dir(settings), debug)
            result.fingerprinter = fingerprinter
            return result

    @classmethod
    def from_crawler(cls, crawler):
        try:
            return cls.from_settings(crawler.settings,
                                      fingerprinter=crawler.request_fingerprinter)
        except TypeError:
            warn("DupeFilter subclasses must either modify their overridden "
                 "'__init__' method and 'from_settings' class method to "
                 "support a 'fingerprinter' parameter, or reimplement the "
                 "'from_crawler' class method.", ScrapyDeprecationWarning)
            result = cls.from_settings(crawler.settings)
            result.fingerprinter = crawler.request_fingerprinter
            return result

    def is_duplicate(self, request: Request) -> bool:
        fingerprint = self._get_request_fingerprint(request)
        if fingerprint in self.fingerprints:
            return True
        self.fingerprints.add(fingerprint)
        if self.file:
            self.file.write(fingerprint + '\n')
        return False

    def _get_request_fingerprint(self, request: Request) -> str:
        return self.fingerprinter.fingerprint(request).hex()

    def close(self, reason: str) -> None:
        if self.file:
            self.file.close()

    def log(self, request: Request, spider: Spider) -> None:
        if self.debug:
            msg = 'Filtered duplicate request: %(request)s (referer: %(referer)s)'
            args = {'request': request, 'referer': referer_str(request)}
            self.logger.debug(msg, args, extra={'spider': spider})
        elif self.log_duplicates:
            msg = ('Filtered duplicate request: %(request)s'
                   ' - no more duplicates will be shown'
                   ' (see DUPEFILTER_DEBUG to show all duplicates)')
            self.logger.debug(msg, {'request': request}, extra={'spider': spider})
            self.log_duplicates = False

        spider.crawler.stats.inc_value('dupefilter/filtered', spider=spider)