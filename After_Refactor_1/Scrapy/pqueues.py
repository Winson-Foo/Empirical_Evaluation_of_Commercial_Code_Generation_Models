import hashlib
import logging
from typing import Any, List, Tuple

from scrapy.utils.misc import create_instance


logger = logging.getLogger(__name__)


def _path_safe(text: str) -> str:
    """
    Return a filesystem-safe version of a string ``text``
    """
    pathable_slot = "".join([c if c.isalnum() or c in "-._" else "_" for c in text])
    unique_slot = hashlib.md5(text.encode("utf8")).hexdigest()
    return "-".join([pathable_slot, unique_slot])


class ScrapyPriorityQueue:
    """
    A priority queue implemented using multiple internal queues (typically,
    FIFO queues).
    """

    @classmethod
    def from_crawler(cls, crawler: Any, downstream_queue_cls: Any, key: str, startprios: List[int] = []) -> Any:
        return cls(crawler, downstream_queue_cls, key, startprios)

    def __init__(self, crawler: Any, downstream_queue_cls: Any, key: str, startprios: List[int] = []) -> None:
        self.crawler = crawler
        self.downstream_queue_cls = downstream_queue_cls
        self.key = key
        self.queues = {}
        self.curprio = None
        self.init_prios(startprios)

    def init_prios(self, startprios: List[int]) -> None:
        """
        Initialize the priority queue with the given priorities.
        """
        if not startprios:
            return

        for priority in startprios:
            self.queues[priority] = self.qfactory(priority)

        self.curprio = min(startprios)

    def qfactory(self, key: int) -> Any:
        """
        Create a new internal queue with the given priority key.
        """
        return create_instance(
            self.downstream_queue_cls,
            None,
            self.crawler,
            self.key + "/" + str(key),
        )

    def priority(self, request: Any) -> int:
        """
        Get the priority for a given request.
        """
        return -request.priority

    def push(self, request: Any) -> None:
        """
        Add a request to the priority queue.
        """
        priority = self.priority(request)
        if priority not in self.queues:
            self.queues[priority] = self.qfactory(priority)
        queue = self.queues[priority]
        queue.push(request)
        if self.curprio is None or priority < self.curprio:
            self.curprio = priority

    def pop(self) -> Any:
        """
        Remove and return the next request from the priority queue.
        """
        if self.curprio is None:
            return None
        queue = self.queues[self.curprio]
        request = queue.pop()
        if not queue:
            del self.queues[self.curprio]
            queue.close()
            prios = [p for p, q in self.queues.items() if q]
            self.curprio = min(prios) if prios else None
        return request

    def peek(self) -> Any:
        """
        Get the next request in the priority queue without removing it.
        """
        if self.curprio is None:
            return None
        queue = self.queues[self.curprio]
        return queue.peek()

    def close(self) -> None:
        """
        Close the priority queue.
        """
        active = []
        for p, q in self.queues.items():
            active.append(p)
            q.close()
        return active

    def __len__(self) -> int:
        """
        Get the number of requests in the priority queue.
        """
        return sum(len(x) for x in self.queues.values()) if self.queues else 0


class DownloaderInterface:
    """
    Interface for accessing the Scrapy downloader.
    """

    def __init__(self, crawler: Any) -> None:
        self.downloader = crawler.engine.downloader

    def stats(self, possible_slots: List[str]) -> List[Tuple[int, str]]:
        """
        Get the number of requests currently being processed by the Scrapy downloader
        for the given slots.
        """
        return [
            (self._active_downloads(slot), slot) for slot in possible_slots
        ]

    def get_slot_key(self, request: Any) -> str:
        """
        Get the slot key for a given request.
        """
        return self.downloader._get_slot_key(request, None)

    def _active_downloads(self, slot: str) -> int:
        """
        Get the number of active downloads for a given slot.
        """
        if slot not in self.downloader.slots:
            return 0
        return len(self.downloader.slots[slot].active)


class DownloaderAwarePriorityQueue:
    """
    PriorityQueue which takes Downloader activity into account:
    domains (slots) with the least amount of active downloads are dequeued first.
    """

    @classmethod
    def from_crawler(cls, crawler: Any, downstream_queue_cls: Any, key: str, startprios: List[int] = []) -> Any:
        return cls(crawler, downstream_queue_cls, key, startprios)

    def __init__(self, crawler: Any, downstream_queue_cls: Any, key: str, slot_startprios: List[int] = []) -> None:
        if crawler.settings.getint("CONCURRENT_REQUESTS_PER_IP") != 0:
            raise ValueError(
                f'"{self.__class__}" does not support CONCURRENT_REQUESTS_PER_IP'
            )

        if slot_startprios and not isinstance(slot_startprios, dict):
            raise ValueError(
                "DownloaderAwarePriorityQueue accepts "
                "``slot_startprios`` as a dict; "
                f"{slot_startprios.__class__!r} instance "
                "is passed. Most likely, it means the state is"
                "created by an incompatible priority "
                "queue. Only a crawl started with the same priority "
                "queue class can be resumed."
            )

        self._downloader_interface = DownloaderInterface(crawler)
        self.downstream_queue_cls = downstream_queue_cls
        self.key = key
        self.crawler = crawler
        self.pqueues = {}  # slot -> priority queue

        for slot, startprios in (slot_startprios or {}).items():
            self.pqueues[slot] = self.pqfactory(slot, startprios)

    def pqfactory(self, slot: str, startprios: List[int] = []) -> Any:
        """
        Create a new priority queue for the given slot.
        """
        return ScrapyPriorityQueue(
            self.crawler,
            self.downstream_queue_cls,
            self.key + "/" + _path_safe(slot),
            startprios,
        )

    def pop(self) -> Any:
        """
        Remove and return the next request from the Deque with least amount of downloads.
        """
        stats = self._downloader_interface.stats(list(self.pqueues.keys()))

        if not stats:
            return None

        slot = min(stats)[1]
        queue = self.pqueues[slot]
        request = queue.pop()
        if len(queue) == 0:
            del self.pqueues[slot]
        return request

    def push(self, request: Any) -> None:
        """
        Add a request to the appropriate priority queue.
        """
        slot = self._downloader_interface.get_slot_key(request)
        if slot not in self.pqueues:
            self.pqueues[slot] = self.pqfactory(slot)
        queue = self.pqueues[slot]
        queue.push(request)

    def peek(self) -> Any:
        """
        Get the next request in the queue without removing it.
        """
        stats = self._downloader_interface.stats(list(self.pqueues.keys()))
        if not stats:
            return None
        slot = min(stats)[1]
        queue = self.pqueues[slot]
        return queue.peek()

    def close(self) -> None:
        """
        Close the priority queues.
        """
        active = {slot: queue.close() for slot, queue in self.pqueues.items()}
        self.pqueues.clear()
        return active

    def __len__(self) -> int:
        """
        Get the number of requests in the priority queues.
        """
        return sum(len(x) for x in self.pqueues.values()) if self.pqueues else 0

    def __contains__(self, slot: str) -> bool:
        """
        Returns True if the given slot is in the priority queues, otherwise False.
        """
        return slot in self.pqueues 