import logging
import os
from pathlib import Path
from typing import Optional, Set

from scrapy.http import Request
from scrapy.settings import BaseSettings
from scrapy.utils.deprecate import ScrapyDeprecationWarning
from scrapy.utils.request import RequestFingerprinter, referer_str


class BaseDupeFilter:
    @classmethod
    def from_settings(cls, settings: BaseSettings) -> "BaseDupeFilter":
        return cls()

    def request_seen(self, request: Request) -> bool:
        return False

    def open(self) -> Optional[os.sys]:
        pass

    def close(self, reason: str) -> Optional[os.sys]:
        pass

    def log(self, request: Request, spider: "Spider") -> None:
        pass


class RFPDupeFilter(BaseDupeFilter):
    def __init__(
        self,
        file_path: Optional[str] = None,
        debug_mode: bool = False,
        fingerprinter: RequestFingerprinter = None,
    ):
        self.file = None
        self.fingerprinter = fingerprinter or RequestFingerprinter()
        self.fingerprints: Set[str] = set()
        self.log_dupes = True
        self.debug_mode = debug_mode
        self.logger = logging.getLogger(__name__)
        if file_path:
            self.file = Path(file_path, "requests.seen").open("a+", encoding="utf-8")
            self.file.seek(0)
            self.fingerprints.update(line.rstrip() for line in self.file)

    @classmethod
    def from_settings(
        cls, settings: BaseSettings, fingerprinter: RequestFingerprinter = None
    ) -> "RFPDupeFilter":
        debug_mode = settings.getbool("DUPEFILTER_DEBUG")
        try:
            return cls(
                file_path=settings.get("JOBDIR"),
                debug_mode=debug_mode,
                fingerprinter=fingerprinter,
            )
        except TypeError:
            warn(
                "RFPDupeFilter subclasses must either modify their '__init__' "
                "method to support a 'fingerprinter' parameter or reimplement "
                "the 'from_settings' class method.",
                ScrapyDeprecationWarning,
            )
            result = cls(file_path=settings.get("JOBDIR"), debug_mode=debug_mode)
            result.fingerprinter = fingerprinter
            return result

    @classmethod
    def from_crawler(cls, crawler) -> "RFPDupeFilter":
        try:
            return cls.from_settings(
                settings=crawler.settings,
                fingerprinter=crawler.request_fingerprinter,
            )
        except TypeError:
            warn(
                "RFPDupeFilter subclasses must either modify their overridden "
                "'__init__' method and 'from_settings' class method to "
                "support a 'fingerprinter' parameter, or reimplement the "
                "'from_crawler' class method.",
                ScrapyDeprecationWarning,
            )
            result = cls.from_settings(settings=crawler.settings)
            result.fingerprinter = crawler.request_fingerprinter
            return result

    def request_seen(self, request: Request) -> bool:
        fingerprint = self.request_fingerprint(request)
        if fingerprint in self.fingerprints:
            return True
        self.fingerprints.add(fingerprint)
        if self.file:
            self.file.write(fingerprint + "\n")
        return False

    def request_fingerprint(self, request: Request) -> str:
        return self.fingerprinter.fingerprint(request).hex()

    def close(self, reason: str) -> None:
        if self.file:
            self.file.close()

    def log(self, request: Request, spider: "Spider") -> None:
        if self.debug_mode:
            msg = (
                "Filtered duplicate request: %(request)s (referer: %(referer)s)"
            )
            args = {"request": request, "referer": referer_str(request)}
            self.logger.debug(msg, args, extra={"spider": spider})
        elif self.log_dupes:
            msg = (
                "Filtered duplicate request: %(request)s"
                " - no more duplicates will be shown"
                " (see DUPEFILTER_DEBUG to show all duplicates)"
            )
            self.logger.debug(msg, {"request": request}, extra={"spider": spider})
            self.log_dupes = False

        spider.crawler.stats.inc_value("dupefilter/filtered", spider=spider)