import logging
import pprint
from typing import Dict, Optional, Union

from scrapy import Spider, Crawler


logger = logging.getLogger(__name__)


StatsT = Dict[str, Union[str, int, float]]


class StatsCollector:
    def __init__(self, crawler: Crawler) -> None:
        self._dump: bool = crawler.settings.getbool("STATS_DUMP")
        self._stats: StatsT = {}

    def get_value(
        self, key: str, default: Optional[str] = None, spider: Optional[Spider] = None
    ) -> Optional[str]:
        return self._stats.get(key, default)

    def get_stats(self, spider: Optional[Spider] = None) -> StatsT:
        return self._stats

    def set_value(self, key: str, value: str, spider: Optional[Spider] = None) -> None:
        self._stats[key] = value

    def inc_value(
        self, key: str, count: int = 1, start: int = 0, spider: Optional[Spider] = None
    ) -> None:
        self._stats[key] = self._stats.setdefault(key, start) + count

    def dec_value(
        self, key: str, count: int = 1, start: int = 0, spider: Optional[Spider] = None
    ) -> None:
        self.inc_value(key, -count, start, spider)

    def max_value(self, key: str, value: Union[int, float], spider: Optional[Spider] = None) -> None:
        self._stats[key] = max(self._stats.setdefault(key, value), value)

    def min_value(self, key: str, value: Union[int, float], spider: Optional[Spider] = None) -> None:
        self._stats[key] = min(self._stats.setdefault(key, value), value)

    def clear_stats(self, spider: Optional[Spider] = None) -> None:
        self._stats.clear()

    def open_spider(self, spider: Spider) -> None:
        pass

    def close_spider(self, spider: Spider, reason: str) -> None:
        if self._dump:
            logger.info(
                "Dumping Scrapy stats:\n" + pprint.pformat(self._stats),
                extra={"spider": spider},
            )
        self._persist_stats(self._stats, spider)

    def _persist_stats(self, stats: StatsT, spider: Spider) -> None:
        pass


class MemoryStatsCollector(StatsCollector):
    def __init__(self, crawler: Crawler) -> None:
        super().__init__(crawler)
        self.spider_stats: Dict[str, StatsT] = {}

    def _persist_stats(self, stats: StatsT, spider: Spider) -> None:
        self.spider_stats[spider.name] = stats


class DummyStatsCollector(StatsCollector):
    def get_value(
        self, key: str, default: Optional[str] = None, spider: Optional[Spider] = None
    ) -> Optional[str]:
        return default

    def set_value(self, key: str, value: str, spider: Optional[Spider] = None) -> None:
        pass

    def inc_value(
        self, key: str, count: int = 1, start: int = 0, spider: Optional[Spider] = None
    ) -> None:
        pass

    def max_value(self, key: str, value: Union[int, float], spider: Optional[Spider] = None) -> None:
        pass

    def min_value(self, key: str, value: Union[int, float], spider: Optional[Spider] = None) -> None:
        pass 