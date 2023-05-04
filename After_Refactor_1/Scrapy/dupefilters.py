import logging
from pathlib import Path
from typing import Optional, Set, Type, Union

from scrapy.http.request import Request
from scrapy.settings import BaseSettings
from scrapy.spiders import Spider
from scrapy.utils.deprecate import ScrapyDeprecationWarning
from scrapy.utils.job import job_dir
from scrapy.utils.request import RequestFingerprinter, referer_str

class BaseDupeFilter:

    @classmethod
    def from_settings(cls, settings: BaseSettings) -> "BaseDupeFilter":
        return cls()

    def request_seen(self, request: Request) -> bool:
        return False

    def open(self) -> Optional["Deferred"]:
        pass

    def close(self, reason: str) -> Optional["Deferred"]:
        pass

    def log(self, request: Request, spider: Spider) -> None:
        pass

class RequestFingerprintDupeFilter(BaseDupeFilter):

    def __init__(
        self,
        filepath: Optional[Union[str, Path]] = None,
        debug: bool = False,
        fingerprinter=None,
    ) -> None:
        self.filepath = filepath
        self.fingerprinter = fingerprinter or RequestFingerprinter()
        self.fingerprints: Set[str] = set()
        self.logdupes = True
        self.debug = debug
        self.logger = logging.getLogger(__name__)

        if filepath:
            self.file = Path(filepath, "requests.seen").open("a+", encoding="utf-8")
            self.file.seek(0)
            self.fingerprints.update(x.rstrip() for x in self.file)

    @classmethod
    def from_settings(
        cls,
        settings: BaseSettings,
        fingerprinter=None,
    ) -> "RequestFingerprintDupeFilter":
        debug = settings.getbool("DUPEFILTER_DEBUG")
        filepath = job_dir(settings)

        try:
            return cls(filepath, debug, fingerprinter=fingerprinter)
        except TypeError:
            warn("RFPDupeFilter subclasses must either modify their '__init__' "
                "method to support a 'fingerprinter' parameter or reimplement "
                "'from_settings' class method.", ScrapyDeprecationWarning,)
            
            result = cls(filepath, debug)
            result.fingerprinter = fingerprinter
            return result

    @classmethod
    def from_crawler(cls, crawler: Spider) -> "RequestFingerprintDupeFilter":
        settings = crawler.settings

        try:
            return cls.from_settings(
                settings,
                fingerprinter=crawler.request_fingerprinter,
            )
        except TypeError:
            warn("RFPDupeFilter subclasses must either modify their overridden "
                "'__init__' method and 'from_settings' class method to "
                "support a 'fingerprinter' parameter, or reimplement the "
                "'from_crawler' class method.", ScrapyDeprecationWarning,)
            
            result = cls.from_settings(settings)
            result.fingerprinter = crawler.request_fingerprinter
            return result

    def request_seen(self, request: Request) -> bool:
        fingerprint = self.request_fingerprint(request)

        if fingerprint in self.fingerprints:
            return True

        self.fingerprints.add(fingerprint)

        if self.filepath:
            self.file.write(fingerprint + "\n")

        return False

    def request_fingerprint(self, request: Request) -> str:
        return self.fingerprinter.fingerprint(request).hex()

    def close(self, reason: str) -> None:
        if self.filepath:
            self.file.close()

    def log(self, request: Request, spider: Spider) -> None:
        if self.debug:
            msg = "Filtered duplicate request: %(request)s (referer: %(referer)s)"
            args = {"request": request, "referer": referer_str(request)}
            self.logger.debug(msg, args, extra={"spider": spider})
        elif self.logdupes:
            msg = (
                "Filtered duplicate request: %(request)s"
                " - no more duplicates will be shown"
                " (see DUPEFILTER_DEBUG to show all duplicates)"
            )
            self.logger.debug(msg, {"request": request}, extra={"spider": spider})
            self.logdupes = False

        spider.crawler.stats.inc_value("dupefilter/filtered", spider=spider)