import logging
import pprint
from typing import Any, Dict, Optional

from scrapy import Spider, Crawler

logger = logging.getLogger(__name__)

Stats = Dict[str, Any]


class StatsCollector:
    def __init__(self, crawler: Crawler):
        self.should_dump_stats: bool = crawler.settings.getbool("STATS_DUMP")
        self.total_stats: Stats = {}

    def get_value(self, key: str, default: Any = None, spider: Optional[Spider] = None) -> Any:
        return self.total_stats.get(key, default)

    def get_stats(self, spider: Optional[Spider] = None) -> Stats:
        return self.total_stats

    def set(self, data: Dict[str, Any], spider: Optional[Spider] = None) -> None:
        self.total_stats.update(data)

    def inc(self, key: str, count: int = 1, start: int = 0, spider: Optional[Spider] = None) -> None:
        self.total_stats[key] = self.total_stats.get(key, start) + count

    def max(self, key: str, value: Any, spider: Optional[Spider] = None) -> None:
        self.total_stats[key] = max(self.total_stats.get(key, value), value)

    def min(self, key: str, value: Any, spider: Optional[Spider] = None) -> None:
        self.total_stats[key] = min(self.total_stats.get(key, value), value)

    def clear(self, spider: Optional[Spider] = None) -> None:
        self.total_stats.clear()

    def open_spider(self, spider: Spider) -> None:
        pass

    def close_spider(self, spider: Spider, reason: str) -> None:
        if self.should_dump_stats:
            logger.info(
                "Dumping Scrapy stats:\n" + pprint.pformat(self.total_stats),
                extra={"spider": spider},
            )
        self.persist_stats(self.total_stats, spider)

    def persist_stats(self, stats: Stats, spider: Spider) -> None:
        pass


class MemoryStatsCollector(StatsCollector):
    def __init__(self, crawler: Crawler):
        super().__init__(crawler)
        self.spider_stats: Dict[str, Stats] = {}

    def persist_stats(self, stats: Stats, spider: Spider) -> None:
        self.spider_stats[spider.name] = stats


class DummyStatsCollector(StatsCollector):
    def get_value(self, key: str, default: Any = None, spider: Optional[Spider] = None) -> Any:
        return default

    def set(self, data: Dict[str, Any], spider: Optional[Spider] = None) -> None:
        pass

    def inc(self, key: str, count: int = 1, start: int = 0, spider: Optional[Spider] = None) -> None:
        pass

    def max(self, key: str, value: Any, spider: Optional[Spider] = None) -> None:
        pass

    def min(self, key: str, value: Any, spider: Optional[Spider] = None) -> None:
        pass

    def clear(self, spider: Optional[Spider] = None) -> None:
        pass 