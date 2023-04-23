"""
Scheduler queues.
"""

from pathlib import Path
from typing import Any, Optional, Type, Union

from queuelib import queue

from scrapy.utils.deprecate import create_deprecated_class
from scrapy.utils.request import request_from_dict


def create_directories(queue_class: Type[queue.Queue]) -> Type[queue.Queue]:
    """A decorator that creates the parent directories if they don't exist."""
    class DirectoriesCreated(queue_class):
        def __init__(self, path: Union[str, Path], *args: Any, **kwargs: Any) -> None:
            dirname = Path(path).parent
            if not dirname.exists():
                dirname.mkdir(parents=True, exist_ok=True)
            super().__init__(path, *args, **kwargs)

    return DirectoriesCreated


def make_serializable(queue_class: Type[queue.Queue], serialize_func: callable, deserialize_func: callable) -> Type[queue.Queue]:
    """A decorator that makes a queue serializable."""
    class SerializableQueue(queue_class):
        def push(self, obj: Any) -> None:
            serialized = serialize_func(obj)
            super().push(serialized)

        def pop(self) -> Optional[Any]:
            serialized = super().pop()
            if serialized:
                return deserialize_func(serialized)

        def peek(self) -> Optional[Any]:
            """Returns the next object to be returned by :meth:`pop`,
            but without removing it from the queue.

            Raises :exc:`NotImplementedError` if the underlying queue class does
            not implement a ``peek`` method, which is optional for queues.
            """
            try:
                serialized = super().peek()
            except AttributeError as ex:
                raise NotImplementedError(
                    "The underlying queue class does not implement 'peek'"
                ) from ex
            if serialized:
                return deserialize_func(serialized)

    return SerializableQueue


def make_scrapy_serializable(queue_class: Type[queue.Queue]) -> Type[queue.Queue]:
    """A decorator that makes a queue serializable with Scrapy objects."""
    class ScrapyRequestQueue(queue_class):
        def __init__(self, crawler, key):
            self.spider = crawler.spider
            super().__init__(key)

        @classmethod
        def from_crawler(cls, crawler, key, *args, **kwargs):
            return cls(crawler, key)

        def push(self, obj: Any) -> None:
            obj = obj.to_dict(spider=self.spider)
            super().push(obj)

        def pop(self) -> Optional[Any]:
            obj = super().pop()
            if not obj:
                return None
            return request_from_dict(obj, spider=self.spider)

        def peek(self) -> Optional[Any]:
            """Returns the next object to be returned by :meth:`pop`,
            but without removing it from the queue.

            Raises :exc:`NotImplementedError` if the underlying queue class does
            not implement a ``peek`` method, which is optional for queues.
            """
            obj = super().peek()
            if not obj:
                return None
            return request_from_dict(obj, spider=self.spider)

    return ScrapyRequestQueue


def make_scrapy_non_serializable(queue_class: Type[queue.Queue]) -> Type[queue.Queue]:
    """A decorator that makes a non-serializable Scrapy queue."""
    class ScrapyRequestQueue(queue_class):
        @classmethod
        def from_crawler(cls, crawler, *args, **kwargs):
            return cls()

        def peek(self) -> Optional[Any]:
            """Returns the next object to be returned by :meth:`pop`,
            but without removing it from the queue.

            Raises :exc:`NotImplementedError` if the underlying queue class does
            not implement a ``peek`` method, which is optional for queues.
            """
            try:
                obj = super().peek()
            except AttributeError as ex:
                raise NotImplementedError(
                    "The underlying queue class does not implement 'peek'"
                ) from ex
            return obj

    return ScrapyRequestQueue


def pickle_serialize(obj: Any) -> bytes:
    """Serializes an object using pickle."""
    return pickle.dumps(obj, protocol=4)


PickleFifoSerializationDiskQueue = make_serializable(
    create_directories(queue.FifoDiskQueue), pickle_serialize, pickle.loads
)
PickleLifoSerializationDiskQueue = make_serializable(
    create_directories(queue.LifoDiskQueue), pickle_serialize, pickle.loads
)
MarshalFifoSerializationDiskQueue = make_serializable(
    create_directories(queue.FifoDiskQueue), marshal.dumps, marshal.loads
)
MarshalLifoSerializationDiskQueue = make_serializable(
    create_directories(queue.LifoDiskQueue), marshal.dumps, marshal.loads
)

# public queue classes
PickleFifoDiskQueue = make_scrapy_serializable(PickleFifoSerializationDiskQueue)
PickleLifoDiskQueue = make_scrapy_serializable(PickleLifoSerializationDiskQueue)
MarshalFifoDiskQueue = make_scrapy_serializable(MarshalFifoSerializationDiskQueue)
MarshalLifoDiskQueue = make_scrapy_serializable(MarshalLifoSerializationDiskQueue)
FifoMemoryQueue = make_scrapy_non_serializable(queue.FifoMemoryQueue)
LifoMemoryQueue = make_scrapy_non_serializable(queue.LifoMemoryQueue)


# deprecated queue classes
_subclass_warn_message = "{cls} inherits from deprecated class {old}"
_instance_warn_message = "{cls} is deprecated"
PickleFifoDiskQueueNonRequest = create_deprecated_class(
    name="PickleFifoDiskQueueNonRequest",
    new_class=PickleFifoSerializationDiskQueue,
    subclass_warn_message=_subclass_warn_message,
    instance_warn_message=_instance_warn_message,
)
PickleLifoDiskQueueNonRequest = create_deprecated_class(
    name="PickleLifoDiskQueueNonRequest",
    new_class=PickleLifoSerializationDiskQueue,
    subclass_warn_message=_subclass_warn_message,
    instance_warn_message=_instance_warn_message,
)
MarshalFifoDiskQueueNonRequest = create_deprecated_class(
    name="MarshalFifoDiskQueueNonRequest",
    new_class=MarshalFifoSerializationDiskQueue,
    subclass_warn_message=_subclass_warn_message,
    instance_warn_message=_instance_warn_message,
)
MarshalLifoDiskQueueNonRequest = create_deprecated_class(
    name="MarshalLifoDiskQueueNonRequest",
    new_class=MarshalLifoSerializationDiskQueue,
    subclass_warn_message=_subclass_warn_message,
    instance_warn_message=_instance_warn_message,
)