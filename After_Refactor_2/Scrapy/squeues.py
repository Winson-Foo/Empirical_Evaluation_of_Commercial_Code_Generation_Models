from queuelib import queue
from os import PathLike
from pathlib import Path
from typing import Union
from scrapy.utils.deprecate import create_deprecated_class
from scrapy.utils.request import request_from_dict
import pickle
import marshal


def create_directories(path: Union[str, PathLike], *args, **kwargs):
    '''
    Helper function that creates directories if they don't already exist
    '''
    dirname = Path(path).parent
    if not dirname.exists():
        dirname.mkdir(parents=True, exist_ok=True)
        
    return queue_class(path, *args, **kwargs)


def with_mkdir(queue_class):
    '''
    Decorator function that creates directories when necessary
    '''
    class DirectoriesCreated(queue_class):
        def __init__(self, path: Union[str, PathLike], *args, **kwargs):
            super().__init__(create_directories(path, *args, **kwargs))

    return DirectoriesCreated

        
def serializable_queue(queue_class, serialize, deserialize):
    '''
    Decorator function that serializes and deserializes objects
    '''
    class SerializableQueue(queue_class):
        def push(self, obj):
            s = serialize(obj)
            super().push(s)

        def pop(self):
            s = super().pop()
            if s:
                return deserialize(s)

        def peek(self):
            try:
                s = super().peek()
            except AttributeError as ex:
                raise NotImplementedError(
                    "The underlying queue class does not implement 'peek'"
                ) from ex
            if s:
                return deserialize(s)

    return SerializableQueue


def scrapy_non_serialization_queue(queue_class):
    '''
    Decorator function for non-serializable scrapy request queue
    '''
    class ScrapyRequestQueue(queue_class):
        @classmethod
        def from_crawler(cls, crawler, *args, **kwargs):
            return cls()

        def peek(self):
            try:
                s = super().peek()
            except AttributeError as ex:
                raise NotImplementedError(
                    "The underlying queue class does not implement 'peek'"
                ) from ex
            return s

    return ScrapyRequestQueue


def scrapy_serialization_queue(queue_class):
    '''
    Decorator function for serializable scrapy request queue
    '''
    class ScrapyRequestQueue(queue_class):
        def __init__(self, crawler, key):
            self.spider = crawler.spider
            super().__init__(key)

        @classmethod
        def from_crawler(cls, crawler, key, *args, **kwargs):
            return cls(crawler, key)

        def push(self, request):
            request = request.to_dict(spider=self.spider)
            return super().push(request)

        def pop(self):
            request = super().pop()
            if not request:
                return None
            return request_from_dict(request, spider=self.spider)

        def peek(self):
            try:
                s = super().peek()
            except AttributeError as ex:
                raise NotImplementedError(
                    "The underlying queue class does not implement 'peek'"
                ) from ex
            if not s:
                return None
            return request_from_dict(s, spider=self.spider)

    return ScrapyRequestQueue


def pickle_serialize(obj):
    '''
    Serializes an object using pickle
    '''
    try:
        return pickle.dumps(obj, protocol=4)
    except (pickle.PicklingError, AttributeError, TypeError) as e:
        raise ValueError(str(e)) from e


def marshal_serialize(obj):
    '''
    Serializes an object using marshal
    '''
    try:
        return marshal.dumps(obj)
    except (pickle.PicklingError, AttributeError, TypeError) as e:
        raise ValueError(str(e)) from e


PickleFifoSerializationDiskQueue = serializable_queue(with_mkdir(queue.FifoDiskQueue), pickle_serialize, pickle.loads)
PickleLifoSerializationDiskQueue = serializable_queue(with_mkdir(queue.LifoDiskQueue), pickle_serialize, pickle.loads)
MarshalFifoSerializationDiskQueue = serializable_queue(with_mkdir(queue.FifoDiskQueue), marshal_serialize, marshal.loads)
MarshalLifoSerializationDiskQueue = serializable_queue(with_mkdir(queue.LifoDiskQueue), marshal_serialize, marshal.loads)

FifoMemoryQueue = scrapy_non_serialization_queue(queue.FifoMemoryQueue)
LifoMemoryQueue = scrapy_non_serialization_queue(queue.LifoMemoryQueue)

PickleFifoDiskQueue = scrapy_serialization_queue(PickleFifoSerializationDiskQueue)
PickleLifoDiskQueue = scrapy_serialization_queue(PickleLifoSerializationDiskQueue)
MarshalFifoDiskQueue = scrapy_serialization_queue(MarshalFifoSerializationDiskQueue)
MarshalLifoDiskQueue = scrapy_serialization_queue(MarshalLifoSerializationDiskQueue)

subclass_warn_message = "{cls} inherits from deprecated class {old}"
instance_warn_message = "{cls} is deprecated"

PickleFifoDiskQueueNonRequest = create_deprecated_class(
    name="PickleFifoDiskQueueNonRequest",
    new_class=PickleFifoSerializationDiskQueue,
    subclass_warn_message=subclass_warn_message,
    instance_warn_message=instance_warn_message,
)

PickleLifoDiskQueueNonRequest = create_deprecated_class(
    name="PickleLifoDiskQueueNonRequest",
    new_class=PickleLifoSerializationDiskQueue,
    subclass_warn_message=subclass_warn_message,
    instance_warn_message=instance_warn_message,
)

MarshalFifoDiskQueueNonRequest = create_deprecated_class(
    name="MarshalFifoDiskQueueNonRequest",
    new_class=MarshalFifoSerializationDiskQueue,
    subclass_warn_message=subclass_warn_message,
    instance_warn_message=instance_warn_message,
)

MarshalLifoDiskQueueNonRequest = create_deprecated_class(
    name="MarshalLifoDiskQueueNonRequest",
    new_class=MarshalLifoSerializationDiskQueue,
    subclass_warn_message=subclass_warn_message,
    instance_warn_message=instance_warn_message,
) 