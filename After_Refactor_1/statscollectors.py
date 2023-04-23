"""
Scrapy extension for collecting scraping stats
"""

import logging
import pprint
from typing import Any, Dict, Optional

from scrapy import Spider
from scrapy.crawler import Crawler


logger = logging.getLogger(__name__)
DUMP_STATS = "STATS_DUMP"


class StatsCollector:
    """Class for collecting and managing Scrapy stats."""

    def __init__(self, crawler: Crawler):
        """Initialize the StatsCollector instance.

        Args:
            crawler (Crawler): The Scrapy Crawler instance.
        """
        self.dump_stats: bool = crawler.settings.getbool(DUMP_STATS)
        self.stats: Dict[str, Any] = {}

    def get_value(
        self, key: str, default: Any = None, spider: Optional[Spider] = None
    ) -> Any:
        """Get the value of a specific Scrapy stat.

        Args:
            key (str): The name of the Scrapy stat.
            default (Any, optional): The default value to return if the stat is not found.
                Defaults to None.
            spider (Optional[Spider], optional): The Scrapy spider instance. Defaults to None.

        Returns:
            Any: The value of the requested Scrapy stat or the default value if the stat is not found.
        """
        return self.stats.get(key, default)

    def get_stats(self, spider: Optional[Spider] = None) -> Dict[str, Any]:
        """Get all the Scrapy stats.

        Args:
            spider (Optional[Spider], optional): The Scrapy spider instance. Defaults to None.

        Returns:
            Dict[str, Any]: A dictionary of all Scrapy stats.
        """
        return self.stats

    def set_value(self, key: str, value: Any, spider: Optional[Spider] = None) -> None:
        """Set the value of a specific Scrapy stat.

        Args:
            key (str): The name of the Scrapy stat.
            value (Any): The value of the Scrapy stat.
            spider (Optional[Spider], optional): The Scrapy spider instance. Defaults to None.
        """
        self.stats[key] = value

    def set_stats(self, stats: Dict[str, Any], spider: Optional[Spider] = None) -> None:
        """Set all the Scrapy stats.

        Args:
            stats (Dict[str, Any]): The stats to set.
            spider (Optional[Spider], optional): The Scrapy spider instance. Defaults to None.
        """
        self.stats = stats

    def inc_value(
        self, key: str, count: int = 1, start: int = 0, spider: Optional[Spider] = None
    ) -> None:
        """Increment the value of a specific Scrapy stat.

        Args:
            key (str): The name of the Scrapy stat.
            count (int, optional): The amount to increment the stat by. Defaults to 1.
            start (int, optional): The starting value of the stat. Defaults to 0.
            spider (Optional[Spider], optional): The Scrapy spider instance. Defaults to None.
        """
        self.stats[key] = self.stats.setdefault(key, start) + count

    def max_value(self, key: str, value: Any, spider: Optional[Spider] = None) -> None:
        """Set the maximum value of a specific Scrapy stat.

        Args:
            key (str): The name of the Scrapy stat.
            value (Any): The value to compare with the stat.
            spider (Optional[Spider], optional): The Scrapy spider instance. Defaults to None.
        """
        self.stats[key] = max(self.stats.setdefault(key, value), value)

    def min_value(self, key: str, value: Any, spider: Optional[Spider] = None) -> None:
        """Set the minimum value of a specific Scrapy stat.

        Args:
            key (str): The name of the Scrapy stat.
            value (Any): The value to compare with the stat.
            spider (Optional[Spider], optional): The Scrapy spider instance. Defaults to None.
        """
        self.stats[key] = min(self.stats.setdefault(key, value), value)

    def clear_stats(self, spider: Optional[Spider] = None) -> None:
        """Clear all the Scrapy stats.

        Args:
            spider (Optional[Spider], optional): The Scrapy spider instance. Defaults to None.
        """
        self.stats.clear()

    def open_spider(self, spider: Spider) -> None:
        """Method called when the spider is opened.

        Args:
            spider (Spider): The Scrapy spider instance.
        """
        pass

    def close_spider(self, spider: Spider, reason: str) -> None:
        """Method called when the spider is closed.

        Args:
            spider (Spider): The Scrapy spider instance.
            reason (str): The reason why the spider is closing.
        """
        if self.dump_stats:
            logger.info(
                f"Dumping Scrapy stats:\n {pprint.pformat(self.stats)}", extra={"spider": spider}
            )
        self._persist_stats(self.stats, spider)

    def _persist_stats(self, stats: Dict[str, Any], spider: Spider) -> None:
        """Method to persist the Scrapy stats.

        Args:
            stats (Dict[str, Any]): The Scrapy stats to persist.
            spider (Spider): The Scrapy spider instance.
        """
        pass


class MemoryStatsCollector(StatsCollector):
    """Class for collecting and managing Scrapy stats in memory."""

    def __init__(self, crawler: Crawler):
        """Initialize the MemoryStatsCollector instance.

        Args:
            crawler (Crawler): The Scrapy Crawler instance.
        """
        super().__init__(crawler)
        self.spider_stats: Dict[str, Dict[str, Any]] = {}

    def _persist_stats(self, stats: Dict[str, Any], spider: Spider) -> None:
        """Persist the Scrapy stats in memory.

        Args:
            stats (Dict[str, Any]): The Scrapy stats to persist.
            spider (Spider): The Scrapy spider instance.
        """
        self.spider_stats[spider.name] = stats


class DummyStatsCollector(StatsCollector):
    """Class for a dummy Scrapy stats collector that does nothing."""

    def get_value(
        self, key: str, default: Any = None, spider: Optional[Spider] = None
    ) -> Any:
        """Dummy method that does nothing.

        Returns:
            Any: Returns None.
        """
        return default

    def set_value(self, key: str, value: Any, spider: Optional[Spider] = None) -> None:
        """Dummy method that does nothing."""
        pass

    def set_stats(self, stats: Dict[str, Any], spider: Optional[Spider] = None) -> None:
        """Dummy method that does nothing."""
        pass

    def inc_value(
        self, key: str, count: int = 1, start: int = 0, spider: Optional[Spider] = None
    ) -> None:
        """Dummy method that does nothing."""
        pass

    def max_value(self, key: str, value: Any, spider: Optional[Spider] = None) -> None:
        """Dummy method that does nothing."""
        pass

    def min_value(self, key: str, value: Any, spider: Optional[Spider] = None) -> None:
        """Dummy method that does nothing."""
        pass