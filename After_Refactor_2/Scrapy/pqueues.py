import hashlib
import logging
from typing import List, Tuple, Optional, Dict

from scrapy.utils.misc import create_instance


logger = logging.getLogger(__name__)


def path_safe(text: str) -> str:
    """Return a filesystem-safe version of a string ``text``"""
    pathable_slot = "".join(
        [c if c.isalnum() or c in "-._" else "_" for c in text]
    )
    # as we replace some letters we can get collision for different slots
    # add we add unique part
    unique_slot = hashlib.md5(text.encode("utf8")).hexdigest()
    return "-".join([pathable_slot, unique_slot])


class ScrapyPriorityQueue:
    """A priority queue implemented using multiple internal queues (typically,
    FIFO queues). It uses one internal queue for each priority value. The internal
    queue must implement the following methods:

        * push(obj)
        * pop()
        * close()
        * __len__()

    Optionally, the queue could provide a ``peek`` method, that should return the
    next object to be returned by ``pop``, but without removing it from the queue.
    """

    def __init__(
        self,
        crawler,
        downstream_queue_cls,
        key,
        start_priorities: List[int] = None,
    ):
        self.crawler = crawler
        self.downstream_queue_cls = downstream_queue_cls
        self.key = key
        self.queues = {}
        self.curprio = None
        self.init_priorities(start_priorities or [])

    def init_priorities(self, start_priorities: List[int]):
        for priority in start_priorities:
            self.queues[priority] = self.qfactory(priority)

        self.curprio = min(start_priorities) if start_priorities else None

    def qfactory(self, key):
        return create_instance(
            self.downstream_queue_cls,
            None,
            self.crawler,
            self.key + "/" + str(key),
        )

    def priority(self, request):
        return -request.priority

    def push(self, request):
        priority = self.priority(request)
        if priority not in self.queues:
            self.queues[priority] = self.qfactory(priority)
        queue = self.queues[priority]
        queue.push(request)  # this may fail (eg. serialization error)
        if self.curprio is None or priority < self.curprio:
            self.curprio = priority

    def pop(self):
        if self.curprio is None:
            return
        queue = self.queues[self.curprio]
        item = queue.pop()
        if not queue:
            del self.queues[self.curprio]
            queue.close()
            priorities = [p for p in self.queues.keys() if self.queues[p]]
            self.curprio = min(priorities) if priorities else None
        return item

    def peek(self):
        """Returns the next object to be returned by :meth:`pop`,
        but without removing it from the queue."""
        if self.curprio is None:
            return None
        queue = self.queues[self.curprio]
        return queue.peek() if hasattr(queue, "peek") else None

    def close(self):
        active_priorities = []
        for priority, queue in self.queues.items():
            active_priorities.append(priority)
            queue.close()
        return active_priorities

    def __len__(self):
        return sum(len(queue) for queue in self.queues.values()) if self.queues else 0


class DownloaderInterface:
    def __init__(self, crawler):
        self.downloader = crawler.engine.downloader

    def stats(self, possible_slots: List[str]) -> List[Tuple[int, str]]:
        """Get a list of the number of active requests for each possible slot"""
        return [
            (self._active_downloads(slot), slot)
            for slot in possible_slots
            if slot in self.downloader.slots
        ]

    def get_slot_key(self, request):
        return self.downloader._get_slot_key(request, None)

    def _active_downloads(self, slot):
        """Return the number of active requests for a given slot"""
        return len(self.downloader.slots.get(slot, {}).get("active", []))


class DownloaderAwarePriorityQueue:
    """PriorityQueue which takes Downloader activity into account:
    domains (slots) with the least amount of active downloads are dequeued
    first.
    """

    def __init__(
        self,
        crawler,
        downstream_queue_cls,
        key,
        slot_start_priorities: Dict[str, List[int]] = None,
    ):
        if crawler.settings.getint("CONCURRENT_REQUESTS_PER_IP") != 0:
            raise ValueError(
                f'"{self.__class__}" does not support CONCURRENT_REQUESTS_PER_IP'
            )

        if slot_start_priorities and not isinstance(slot_start_priorities, dict):
            raise ValueError(
                "DownloaderAwarePriorityQueue accepts "
                "``slot_startprios`` as a dict; "
                f"{slot_start_priorities.__class__!r} instance "
                "is passed. Most likely, it means the state is"
                "created by an incompatible priority "
                "queue. Only a crawl started with the same priority "
                "queue class can be resumed."
            )

        self._downloader_interface = DownloaderInterface(crawler)
        self.downstream_queue_cls = downstream_queue_cls
        self.key = key
        self.crawler = crawler

        self.queues = {}
        for slot, start_prios in (slot_start_priorities or {}).items():
            if slot not in self._downloader_interface.downloader.slots:
                continue
            self.queues[slot] = ScrapyPriorityQueue(
                self.crawler,
                self.downstream_queue_cls,
                self.key + "/" + path_safe(slot),
                start_prios,
            )

    def pop(self):
        stats = self._downloader_interface.stats(list(self.queues.keys()))

        if not stats:
            return

        slot_with_fewest_active_requests = min(stats)[1]
        queue = self.queues[slot_with_fewest_active_requests]
        item = queue.pop()
        if not queue:
            del self.queues[slot_with_fewest_active_requests]
        return item

    def push(self, request):
        slot = self._downloader_interface.get_slot_key(request)
        if slot not in self._downloader_interface.downloader.slots:
            return
        if slot not in self.queues:
            self.queues[slot] = ScrapyPriorityQueue(
                self.crawler,
                self.downstream_queue_cls,
                self.key + "/" + path_safe(slot),
            )
        queue = self.queues[slot]
        queue.push(request)

    def peek(self):
        """Returns the next object to be returned by :meth:`pop`,
        but without removing it from the queue."""
        stats = self._downloader_interface.stats(list(self.queues.keys()))
        if not stats:
            return None
        slot_with_fewest_active_requests = min(stats)[1]
        queue = self.queues[slot_with_fewest_active_requests]
        return queue.peek() if hasattr(queue, "peek") else None

    def close(self):
        active_priorities_by_slot = {
            slot: queue.close() for slot, queue in self.queues.items()
        }
        self.queues = {}
        return active_priorities_by_slot

    def __len__(self):
        return sum(len(queue) for queue in self.queues.values()) if self.queues else 0

    def __contains__(self, slot):
        return slot in self.queues 