import hashlib
import logging

from scrapy.utils.misc import create_instance


logger = logging.getLogger(__name__)


class PriorityQueue:
    """
    A priority queue implemented using multiple internal queues (typically, FIFO queues).
    It uses one internal queue for each priority value. The internal queue must implement the following methods:

    * push(obj)
    * pop()
    * close()
    * __len__()

    Optionally, the queue could provide a `peek` method, that should return the next object to be returned by `pop`,
    but without removing it from the queue.

    Only integer priorities should be used. Lower numbers are higher priorities.

    startprios is a sequence of priorities to start with. If the queue was previously closed leaving some priority buckets
    non-empty, those priorities should be passed in startprios.
    """
    @classmethod
    def from_crawler(cls, crawler, downstream_queue_cls, key, startprios=()):
        return cls(crawler, downstream_queue_cls, key, startprios)

    def __init__(self, crawler, downstream_queue_cls, key, startprios=()):
        """
        Initializes the PriorityQueue with the given arguments.
        """
        self.crawler = crawler
        self.downstream_queue_cls = downstream_queue_cls
        self.key = key
        self.queues = {}
        self.curprio = None

        for priority in startprios:
            self.queues[priority] = self.qfactory(priority)

        self.curprio = min(startprios) if startprios else None

    def qfactory(self, key):
        return create_instance(
            self.downstream_queue_cls,
            None,
            self.crawler,
            self.key + "/" + str(key),
        )

    @staticmethod
    def _path_safe(text):
        """
        Returns a filesystem-safe version of a given string.
        """
        # ...
    
    def priority(self, request):
        return -request.priority

    def push(self, request):
        priority = self.priority(request)
        if priority not in self.queues:
            self.queues[priority] = self.qfactory(priority)
        q = self.queues[priority]
        q.push(request)  # this may fail (eg. serialization error)
        if self.curprio is None or priority < self.curprio:
            self.curprio = priority

    def pop(self):
        if self.curprio is None:
            return
        q = self.queues[self.curprio]
        m = q.pop()
        if not q:
            del self.queues[self.curprio]
            q.close()
            prios = [p for p, q in self.queues.items() if q]
            self.curprio = min(prios) if prios else None
        return m

    def peek(self):
        if self.curprio is None:
            return None
        queue = self.queues[self.curprio]
        return queue.peek()

    def close(self):
        active = []
        for p, q in self.queues.items():
            active.append(p)
            q.close()
        return active

    def __len__(self):
        return sum(len(x) for x in self.queues.values()) if self.queues else 0


def _active_downloads(downloader, slot):
    """
    Returns a number of requests in a Downloader for a given slot.
    """
    if slot not in downloader.slots:
        return 0
    return len(downloader.slots[slot].active)


class DownloaderAwarePriorityQueuePriorityQueue:
    """
    PriorityQueue which takes Downloader activity into account:
    domains (slots) with the least amount of active downloads are dequeued first.
    """
    @classmethod
    def from_crawler(cls, crawler, downstream_queue_cls, key, queue_state=()):
        return cls(crawler, downstream_queue_cls, key, queue_state)

    def __init__(self, crawler, downstream_queue_cls, key, queue_state=()):
        """
        Initializes the DownloaderAwarePriorityQueuePriorityQueue with the given arguments.
        """
        if crawler.settings.getint("CONCURRENT_REQUESTS_PER_IP") != 0:
            raise ValueError(
                f"{self.__class__} does not support CONCURRENT_REQUESTS_PER_IP."
            )

        if queue_state and not isinstance(queue_state, dict):
            raise ValueError(
                "DownloaderAwarePriorityQueuePriorityQueue accepts"
                "``queue_state`` as a dict;"
                f"{queue_state.__class__!r} instance is passed. Most likely, it means the state is"
                "created by an incompatible priority queue. Only a crawl started with the same priority "
                "queue class can be resumed."
            )

        self._downloader_interface = DownloaderInterface(crawler)
        self.downstream_queue_cls = downstream_queue_cls
        self.key = key
        self.crawler = crawler

        self.pqueues = {}  # slot -> priority queue
        for slot, startprios in queue_state.items():
            self.pqueues[slot] = self.pqfactory(slot, startprios)

    def pqfactory(self, slot, startprios=()):
        return PriorityQueue(
            self.crawler,
            self.downstream_queue_cls,
            self.key + "/" + _path_safe(slot),
            startprios,
        )

    def pop(self):
        stats = self._downloader_interface.stats(self.pqueues)

        if not stats:
            return

        slot = min(stats)[1]
        queue = self.pqueues[slot]
        request = queue.pop()
        if len(queue) == 0:
            del self.pqueues[slot]
        return request

    def push(self, request):
        slot = self._downloader_interface.get_slot_key(request)
        if slot not in self.pqueues:
            self.pqueues[slot] = self.pqfactory(slot)
        queue = self.pqueues[slot]
        queue.push(request)

    def peek(self):
        stats = self._downloader_interface.stats(self.pqueues)
        if not stats:
            return None
        slot = min(stats)[1]
        queue = self.pqueues[slot]
        return queue.peek()

    def close(self):
        active = {slot: queue.close() for slot, queue in self.pqueues.items()}
        self.pqueues.clear()
        return active

    def __len__(self):
        return sum(len(x) for x in self.pqueues.values()) if self.pqueues else 0

    def __contains__(self, slot):
        return slot in self.pqueues