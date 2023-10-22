from os import PathLike
from pathlib import Path
from typing import Union

import marshal
import pickle
from queuelib import queue

from scrapy.utils.deprecate import create_deprecated_class
from scrapy.utils.request import request_from_dict


def create_directories_if_not_exist(path: Union[str, PathLike], queue_class):
    """
    A helper function to create directories if they do not exist before initializing a queue class.
    """
    class QueueWithDirectoriesCreated(queue_class):
        def __init__(self, path: Union[str, PathLike], *args, **kwargs):
            directory = Path(path).parent
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
            super().__init__(path, *args, **kwargs)

    return QueueWithDirectoriesCreated(path)


def create_serializable_queue(queue_class, serialize_method, deserialize_method):
    """
    A helper function to create a queue class that can serialize and deserialize objects.
    """
    class SerializableQueue(queue_class):
        def push(self, obj):
            serialized_obj = serialize_method(obj)
            super().push(serialized_obj)

        def pop(self):
            serialized_obj = super().pop()
            if serialized_obj:
                return deserialize_method(serialized_obj)

        def peek(self):
            try:
                serialized_obj = super().peek()
            except AttributeError as e:
                raise NotImplementedError(
                    "The underlying queue class does not implement 'peek' method."
                ) from e
            if serialized_obj:
                return deserialize_method(serialized_obj)

    return SerializableQueue(queue_class)


def create_scrapy_serialization_queue(queue_class):
    """
    A helper function to create a queue class that supports serialization for Scrapy.
    """
    class ScrapySerializableQueue(queue_class):
        def __init__(self, crawler, key):
            self.spider = crawler.spider
            super().__init__(key)

        @classmethod
        def from_crawler(cls, crawler, key, *args, **kwargs):
            return cls(crawler, key)

        def push(self, request):
            request_dict = request.to_dict(spider=self.spider)
            return super().push(request_dict)

        def pop(self):
            request_dict = super().pop()
            if not request_dict:
                return None
            return request_from_dict(request_dict, spider=self.spider)

        def peek(self):
            try:
                request_dict = super().peek()
            except AttributeError as e:
                raise NotImplementedError(
                    "The underlying queue class does not implement 'peek' method."
                ) from e
            if request_dict:
                return request_from_dict(request_dict, spider=self.spider)

    return ScrapySerializableQueue(queue_class)


def create_scrapy_non_serialization_queue(queue_class):
    """
    A helper function to create a queue class that does not support serialization for Scrapy.
    """
    class ScrapyNonSerializableQueue(queue_class):
        @classmethod
        def from_crawler(cls, crawler, *args, **kwargs):
            return cls()

        def peek(self):
            try:
                obj = super().peek()
            except AttributeError as e:
                raise NotImplementedError(
                    "The underlying queue class does not implement 'peek' method."
                ) from e
            return obj

    return ScrapyNonSerializableQueue(queue_class)


def serialize_pickle(obj):
    """
    A helper function to serialize an object using the pickle module.
    """
    try:
        return pickle.dumps(obj, protocol=4)
    except (pickle.PicklingError, AttributeError, TypeError) as e:
        raise ValueError(str(e)) from e


def serialize_marshal(obj):
    """
    A helper function to serialize an object using the marshal module.
    """
    return marshal.dumps(obj)


# Queue classes with directories created if they do not exist
PickleFifoDiskQueueWithDirectories = create_directories_if_not_exist(
    'pickle_fifo_disk_queue',
    queue.FifoDiskQueue,
)
PickleLifoDiskQueueWithDirectories = create_directories_if_not_exist(
    'pickle_lifo_disk_queue',
    queue.LifoDiskQueue,
)

MarshalFifoDiskQueueWithDirectories = create_directories_if_not_exist(
    'marshal_fifo_disk_queue',
    queue.FifoDiskQueue,
)

MarshalLifoDiskQueueWithDirectories = create_directories_if_not_exist(
    'marshal_lifo_disk_queue',
    queue.LifoDiskQueue,
)

# Queue classes that support serialization for Scrapy
PickleFifoDiskQueue = create_scrapy_serialization_queue(
    create_serializable_queue(
        PickleFifoDiskQueueWithDirectories,
        serialize_pickle,
        pickle.loads,
    )
)

PickleLifoDiskQueue = create_scrapy_serialization_queue(
    create_serializable_queue(
        PickleLifoDiskQueueWithDirectories,
        serialize_pickle,
        pickle.loads,
    )
)

MarshalFifoDiskQueue = create_scrapy_serialization_queue(
    create_serializable_queue(
        MarshalFifoDiskQueueWithDirectories,
        serialize_marshal,
        marshal.loads,
    )
)

MarshalLifoDiskQueue = create_scrapy_serialization_queue(
    create_serializable_queue(
        MarshalLifoDiskQueueWithDirectories,
        serialize_marshal,
        marshal.loads,
    )
)

FifoMemoryQueue = create_scrapy_non_serialization_queue(queue.FifoMemoryQueue)
LifoMemoryQueue = create_scrapy_non_serialization_queue(queue.LifoMemoryQueue)


"""
Deprecated queue classes
"""

# A warning message for deprecated queue classes
subclass_warn_msg = "{cls} inherits from deprecated class {old}"
instance_warn_msg = "{cls} is deprecated"

# Deprecated queue classes
PickleFifoDiskQueueNonRequest = create_deprecated_class(
    name="PickleFifoDiskQueueNonRequest",
    new_class=PickelFifoDiskQueueWithDirectories,
    subclass_warn_message=subclass_warn_msg,
    instance_warn_message=instance_warn_msg,
)

PickleLifoDiskQueueNonRequest = create_deprecated_class(
    name="PickleLifoDiskQueueNonRequest",
    new_class=PickelLifoDiskQueueWithDirectories,
    subclass_warn_message=subclass_warn_msg,
    instance_warn_message=instance_warn_msg,
)

MarshalFifoDiskQueueNonRequest = create_deprecated_class(
    name="MarshalFifoDiskQueueNonRequest",
    new_class=MarshalFifoDiskQueueWithDirectories,
    subclass_warn_message=subclass_warn_msg,
    instance_warn_message=instance_warn_msg,
)

MarshalLifoDiskQueueNonRequest = create_deprecated_class(
    name="MarshalLifoDiskQueueNonRequest",
    new_class=MarshalLifoDiskQueueWithDirectories,
    subclass_warn_message=subclass_warn_msg,
    instance_warn_message=instance_warn_msg,
) 