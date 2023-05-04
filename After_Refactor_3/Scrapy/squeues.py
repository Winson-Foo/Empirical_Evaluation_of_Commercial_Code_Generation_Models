from os import PathLike
from pathlib import Path
from typing import Any, Type
import marshal

from queuelib import queue

from scrapy.utils.request import request_from_dict


class DirectoriesCreatedQueue(queue.FifoDiskQueue):
    """
    Queue that automatically creates its directory if it doesn't exist.
    """
    def __init__(self, path: Union[str, PathLike], *args: Any, **kwargs: Any) -> None:
        dirname = Path(path).parent
        if not dirname.exists():
            dirname.mkdir(parents=True, exist_ok=True)
        super().__init__(path, *args, **kwargs)


class SerializableQueue(queue.Queue):
    """
    Base class for serializable queues.
    """
    serialize = None
    deserialize = None

    def push(self, obj: Any) -> None:
        s = self.serialize(obj)
        super().push(s)

    def pop(self) -> Any:
        s = super().pop()
        if s:
            return self.deserialize(s)

    def peek(self) -> Any:
        """
        Returns the next object to be returned by :meth:`pop`,
        but without removing it from the queue.

        Raises :exc:`NotImplementedError` if the underlying queue class does
        not implement a `peek` method, which is optional for queues.
        """
        try:
            s = super().peek()
        except AttributeError as ex:
            raise NotImplementedError(
                "The underlying queue class does not implement 'peek'"
            ) from ex
        if s:
            return self.deserialize(s)


class ScrapyRequestQueue(queue.Queue):
    """
    Queue for Scrapy requests.
    """
    def __init__(self, crawler: Any, key: str) -> None:
        self.spider = crawler.spider
        super().__init__(key)

    @classmethod
    def from_crawler(cls, crawler: Any, key: str, *args: Any, **kwargs: Any) -> Type['ScrapyRequestQueue']:
        return cls(crawler, key)

    def push(self, request: Any) -> None:
        request = request.to_dict(spider=self.spider)
        return super().push(request)

    def pop(self) -> Any:
        request = super().pop()
        if not request:
            return None
        return request_from_dict(request, spider=self.spider)

    def peek(self) -> Any:
        """
        Returns the next object to be returned by :meth:`pop`,
        but without removing it from the queue.

        Raises :exc:`NotImplementedError` if the underlying queue class does
        not implement a `peek` method, which is optional for queues.
        """
        request = super().peek()
        if not request:
            return None
        return request_from_dict(request, spider=self.spider)


def create_serializable_queue(queue_class: Type[queue.Queue], serialize: Any, deserialize: Any) -> Type[SerializableQueue]:
    """
    Factory function that creates a serializable queue class for a given queue class.
    """
    new_class = type(f'Serializable{queue_class.__name__}', (SerializableQueue, queue_class), {})
    new_class.serialize = serialize
    new_class.deserialize = deserialize
    return new_class


def create_scrapy_serialization_queue(queue_class: Type[queue.Queue]) -> Type[ScrapyRequestQueue]:
    """
    Factory function that creates a Scrapy serialization queue class for a given queue class.
    """
    return type(f'ScrapySerialization{queue_class.__name__}', (ScrapyRequestQueue, create_serializable_queue(queue_class, pickle.dumps, pickle.loads)), {})


def create_scrapy_non_serialization_queue(queue_class: Type[queue.Queue]) -> Type[ScrapyRequestQueue]:
    """
    Factory function that creates a Scrapy non-serialization queue class for a given queue class.
    """
    return type(f'ScrapyNonSerialization{queue_class.__name__}', (ScrapyRequestQueue, queue_class), {})


PickleFifoDiskQueue = create_scrapy_serialization_queue(DirectoriesCreatedQueue)
PickleLifoDiskQueue = create_scrapy_serialization_queue(DirectoriesCreatedQueue)
MarshalFifoDiskQueue = create_scrapy_serialization_queue(DirectoriesCreatedQueue)
MarshalLifoDiskQueue = create_scrapy_serialization_queue(DirectoriesCreatedQueue)
FifoMemoryQueue = create_scrapy_non_serialization_queue(queue.FifoMemoryQueue)
LifoMemoryQueue = create_scrapy_non_serialization_queue(queue.LifoMemoryQueue)


# Deprecated queue classes
PickleFifoDiskQueueNonRequest = create_deprecated_class(
    name="PickleFifoDiskQueueNonRequest",
    new_class=create_serializable_queue(DirectoriesCreatedQueue, pickle.dumps, pickle.loads),
    subclass_warn_message="{cls} inherits from deprecated class {old}",
    instance_warn_message="{cls} is deprecated",
)
PickleLifoDiskQueueNonRequest = create_deprecated_class(
    name="PickleLifoDiskQueueNonRequest",
    new_class=create_serializable_queue(DirectoriesCreatedQueue, pickle.dumps, pickle.loads),
    subclass_warn_message="{cls} inherits from deprecated class {old}",
    instance_warn_message="{cls} is deprecated",
)
MarshalFifoDiskQueueNonRequest = create_deprecated_class(
    name="MarshalFifoDiskQueueNonRequest",
    new_class=create_serializable_queue(DirectoriesCreatedQueue, marshal.dumps, marshal.loads),
    subclass_warn_message="{cls} inherits from deprecated class {old}",
    instance_warn_message="{cls} is deprecated",
)
MarshalLifoDiskQueueNonRequest = create_deprecated_class(
    name="MarshalLifoDiskQueueNonRequest",
    new_class=create_serializable_queue(DirectoriesCreatedQueue, marshal.dumps, marshal.loads),
    subclass_warn_message="{cls} inherits from deprecated class {old}",
    instance_warn_message="{cls} is deprecated",
) 