import logging
import pprint
from typing import Any, Dict, Optional

from scrapy import Spider
from scrapy.crawler import Crawler


logger = logging.getLogger(__name__)

Stats = Dict[str, Any]


class StatsCollector:
    """
    Base class for collecting and storing scraping stats.
    """

    def __init__(self, crawler: Crawler):
        self.dump = crawler.settings.getbool("STATS_DUMP")
        self.stats: Stats = {}

    def get_value(self, key: str, default: Any = None, spider: Optional[Spider] = None) -> Any:
        """
        Returns the value of a given key in the stats dict.
        """
        return self.stats.get(key, default)

    def get_stats(self, spider: Optional[Spider] = None) -> Stats:
        """
        Returns the stats dict.
        """
        return self.stats

    def set_value(self, key: str, value: Any, spider: Optional[Spider] = None) -> None:
        """
        Sets the value of a given key in the stats dict.
        """
        self.stats[key] = value

    def set_stats(self, stats: Stats, spider: Optional[Spider] = None) -> None:
        """
        Sets the entire stats dict.
        """
        self.stats = stats

    def inc_value(self, key: str, count: int = 1, start: int = 0, spider: Optional[Spider] = None) -> None:
        """
        Increments the value of a given key in the stats dict by a given count.
        """
        self.stats[key] = self.stats.setdefault(key, start) + count

    def max_value(self, key: str, value: Any, spider: Optional[Spider] = None) -> None:
        """
        Sets the value of a given key in the stats dict to the maximum of its current value or a given value.
        """
        self.stats[key] = max(self.stats.setdefault(key, value), value)

    def min_value(self, key: str, value: Any, spider: Optional[Spider] = None) -> None:
        """
        Sets the value of a given key in the stats dict to the minimum of its current value or a given value.
        """
        self.stats[key] = min(self.stats.setdefault(key, value), value)

    def clear_stats(self, spider: Optional[Spider] = None) -> None:
        """
        Clears the stats dict.
        """
        self.stats.clear()

    def open_spider(self, spider: Spider) -> None:
        """
        Called when the spider is opened.
        """
        pass

    def close_spider(self, spider: Spider, reason: str) -> None:
        """
        Called when the spider is closed.
        """
        if self.dump:
            logger.info(f"Dumping Scrapy stats:\n{pprint.pformat(self.stats)}", extra={"spider": spider})
        self._persist_stats(self.stats, spider)

    def _persist_stats(self, stats: Stats, spider: Spider) -> None:
        """
        Persists the stats dict in a subclass-specific way.
        """
        pass


class MemoryStatsCollector(StatsCollector):
    """
    Subclass of StatsCollector that persists the stats in memory.
    """

    def __init__(self, crawler: Crawler):
        super().__init__(crawler)
        self.spider_stats: Dict[str, Stats] = {}

    def _persist_stats(self, stats: Stats, spider: Spider) -> None:
        """
        Stores the stats dict in spider_stats.
        """
        self.spider_stats[spider.name] = stats


class DummyStatsCollector(StatsCollector):
    """
    Subclass of StatsCollector that doesn't collect any stats.
    """

    def get_value(self, key: str, default: Any = None, spider: Optional[Spider] = None) -> Any:
        return default

    def set_value(self, key: str, value: Any, spider: Optional[Spider] = None) -> None:
        pass

    def set_stats(self, stats: Stats, spider: Optional[Spider] = None) -> None:
        pass

    def inc_value(self, key: str, count: int = 1, start: int = 0, spider: Optional[Spider] = None) -> None:
        pass

    def max_value(self, key: str, value: Any, spider: Optional[Spider] = None) -> None:
        pass

    def min_value(self, key: str, value: Any, spider: Optional[Spider] = None) -> None:
        pass 