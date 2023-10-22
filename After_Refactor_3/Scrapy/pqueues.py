import hashlib
import logging
from typing import Dict, List, Optional, Tuple

from scrapy.core.downloader import Downloader
from scrapy.http import Request

logger = logging.getLogger(__name__)


def path_safe(text: str) -> str:
    """
    Return a filesystem-safe version of a string.

    Args:
        text (str): The text to be converted to a filesystem-safe version.

    Returns:
        str: The filesystem-safe version of the input text.
    """
    pathable_text = "".join([c if c.isalnum() or c in "-._" else "_" for c in text])
    unique_part = hashlib.md5(text.encode("utf8")).hexdigest()
    return "-".join([pathable_text, unique_part])


class ScrapyPriorityQueue:
    """A priority queue implemented using multiple internal queues."""

    def __init__(self, crawler, downstream_queue_cls, key, startprios=()):
        self.crawler = crawler
        self.downstream_queue_cls = downstream_queue_cls
        self.key = key
        self.queues = {}
        self.current_priority = None
        self._init_prios(startprios)

    @classmethod
    def from_crawler(
        cls, crawler, downstream_queue_cls, key, startprios=()
    ) -> "ScrapyPriorityQueue":
        """Create a new instance of ScrapyPriorityQueue.

        Args:
            crawler: The Scrapy crawler instance.
            downstream_queue_cls: The downstream queue class.
            key: The key to be used in the queue.
            startprios: A sequence of priorities to start with.

        Returns:
            ScrapyPriorityQueue: A new instance of ScrapyPriorityQueue.
        """
        return cls(crawler, downstream_queue_cls, key, startprios)

    def _init_prios(self, startprios: Tuple[int]) -> None:
        """Initialize the priority queue with startprios.

        Args:
            startprios (Tuple[int]): A sequence of priorities to start with.

        Returns:
            None
        """
        if not startprios:
            return

        for priority in startprios:
            self.queues[priority] = self._create_downstream_queue(priority)

        self.current_priority = min(startprios)

    def _create_downstream_queue(self, priority: int):
        """Create a new downstream queue.

        Args:
            priority (int): The priority for which the downstream queue should be created.

        Returns:
            DownstreamQueue: A new instance of the downstream queue class.
        """
        return create_instance(
            self.downstream_queue_cls,
            None,
            self.crawler,
            self.key + "/" + str(priority),
        )

    def _get_priority(self, request: Request) -> int:
        """Get the priority of a request.

        Args:
            request (Request): The request object for which the priority should be found.

        Returns:
            int: The priority of the request.
        """
        return -request.priority

    def push(self, request: Request) -> None:
        """Add a request to the priority queue.

        Args:
            request (Request): The request object to be added to the queue.

        Returns:
            None
        """
        priority = self._get_priority(request)
        if priority not in self.queues:
            self.queues[priority] = self._create_downstream_queue(priority)
        queue = self.queues[priority]
        queue.push(request)
        if self.current_priority is None or priority < self.current_priority:
            self.current_priority = priority

    def pop(self) -> Optional[Request]:
        """Remove and return the next item from the priority queue.

        Returns:
            Request: The next request object in the queue.
        """
        if self.current_priority is None:
            return None
        queue = self.queues[self.current_priority]
        request = queue.pop()
        if not queue:
            del self.queues[self.current_priority]
            queue.close()
            priorities = [p for p, q in self.queues.items() if q]
            self.current_priority = min(priorities) if priorities else None
        return request

    def peek(self) -> Optional[Request]:
        """Return the next item in the priority queue without removing it.

        Returns:
            Request: The next request object in the queue.
        """
        if self.current_priority is None:
            return None
        queue = self.queues[self.current_priority]
        return queue.peek()

    def close(self) -> List[int]:
        """Close the priority queue.

        Returns:
            List[int]: A list of active priorities.
        """
        active_priorities = []
        for priority, queue in self.queues.items():
            active_priorities.append(priority)
            queue.close()
        return active_priorities

    def __len__(self) -> int:
        """Return the number of items in the priority queue.

        Returns:
            int: The number of items in the priority queue.
        """
        return (
            sum(len(queue) for queue in self.queues.values()) if self.queues else 0
        )


class DownloaderInterface:
    """Interface to interact with the downloader in Scrapy"""

    def __init__(self, crawler):
        self.downloader: Downloader = crawler.engine.downloader

    def stats(self, possible_slots: List[str]) -> List[Tuple[int, str]]:
        """Get the statistics of the downloader for each slot.

        Args:
            possible_slots (List[str]): The list of possible slots.

        Returns:
            List[Tuple[int, str]]: A list of tuples each containing the number of active requests and the slot name.
        """
        return [
            (self._active_downloads(slot), slot) for slot in possible_slots
        ]

    def get_slot_key(self, request: Request) -> str:
        """Get the slot key of a request.

        Args:
            request (Request): The request object for which the slot key should be found.

        Returns:
            str: The slot key of the request.
        """
        return self.downloader._get_slot_key(request, None)

    def _active_downloads(self, slot: str) -> int:
        """Return the number of active requests in a downloader for a given slot.

        Args:
            slot (str): The name of the slot to be checked.

        Returns:
            int: The number of active requests in the downloader for the given slot.
        """
        if slot not in self.downloader.slots:
            return 0
        return len(self.downloader.slots[slot].active)


class DownloaderAwarePriorityQueue:
    """PriorityQueue which takes Downloader activity into account"""

    def __init__(self, crawler, downstream_queue_cls, key, slot_startprios=()):
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
                "created by an incompatible priority queue. "
                "Only a crawl started with the same priority "
                "queue class can be resumed."
            )

        self.crawler = crawler
        self.downstream_queue_cls = downstream_queue_cls
        self.key = key
        self.downloader_interface = DownloaderInterface(crawler)
        self.queues: Dict[str, ScrapyPriorityQueue] = {}
        for slot, startprios in (slot_startprios or {}).items():
            self.queues[slot] = self._create_priority_queue(slot, startprios)

    @classmethod
    def from_crawler(
        cls, crawler, downstream_queue_cls, key, startprios=()
    ) -> "DownloaderAwarePriorityQueue":
        """Create a new instance of DownloaderAwarePriorityQueue.

        Args:
            crawler: The Scrapy crawler instance.
            downstream_queue_cls: The downstream queue class.
            key: The key to be used in the queue.
            startprios: A sequence of priorities to start with.

        Returns:
            DownloaderAwarePriorityQueue: A new instance of DownloaderAwarePriorityQueue.
        """
        return cls(crawler, downstream_queue_cls, key, startprios)

    def _create_priority_queue(self, slot: str, startprios: Tuple[int]):
        """Create a new priority queue.

        Args:
            slot (str): The slot name for which the priority queue should be created.
            startprios (Tuple[int]): A sequence of priorities to start with.

        Returns:
            ScrapyPriorityQueue: A new instance of ScrapyPriorityQueue.
        """
        return ScrapyPriorityQueue(
            self.crawler,
            self.downstream_queue_cls,
            self.key + "/" + path_safe(slot),
            startprios,
        )

    def pop(self) -> Optional[Request]:
        """Remove and return the next item from the priority queue.

        Returns:
            Request: The next request object in the queue.
        """
        stats = self.downloader_interface.stats(list(self.queues.keys()))
        if not stats:
            return None
        slot = min(stats)[1]
        queue = self.queues[slot]
        request = queue.pop()
        if len(queue) == 0:
            del self.queues[slot]
        return request

    def push(self, request: Request) -> None:
        """Add a request to the priority queue.

        Args:
            request (Request): The request object to be added to the queue.

        Returns:
            None
        """
        slot = self.downloader_interface.get_slot_key(request)
        if slot not in self.queues:
            self.queues[slot] = self._create_priority_queue(slot, ())
        queue = self.queues[slot]
        queue.push(request)

    def peek(self) -> Optional[Request]:
        """Return the next item in the priority queue without removing it.

        Returns:
            Request: The next request object in the queue.
        """
        stats = self.downloader_interface.stats(list(self.queues.keys()))
        if not stats:
            return None
        slot = min(stats)[1]
        queue = self.queues[slot]
        return queue.peek()

    def __len__(self) -> int:
        """Return the number of items in the priority queue.

        Returns:
            int: The number of items in the priority queue.
        """
        return sum(len(queue) for queue in self.queues.values()) if self.queues else 0

    def __contains__(self, slot