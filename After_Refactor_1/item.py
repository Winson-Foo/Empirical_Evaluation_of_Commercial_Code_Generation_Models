from copy import copy, deepcopy
from pprint import pformat
from typing import Dict

from scrapy.utils.trackref import object_ref


class Field(dict):
    """Container of field metadata"""


class ItemMeta(type):
    """Metaclass of Item that handles field definitions."""

    def __new__(cls, class_name, bases, attrs):
        fields = attrs.get("fields", {})
        for name, value in attrs.items():
            if isinstance(value, Field):
                fields[name] = value

        attrs["fields"] = fields
        return super().__new__(cls, class_name, bases, attrs)


class Item(MutableMapping, object_ref, metaclass=ItemMeta):
    """
    Base class for scraped items.

    In Scrapy, an object is considered an ``item`` if it is an instance of either
    :class:`Item` or :class:`dict`, or any subclass. For example, when the output of a
    spider callback is evaluated, only instances of :class:`Item` or
    :class:`dict` are passed to :ref:`item pipelines <topics-item-pipeline>`.

    If you need instances of a custom class to be considered items by Scrapy,
    you must inherit from either :class:`Item` or :class:`dict`.

    Items must declare :class:`Field` attributes, which are processed and stored
    in the ``fields`` attribute. This restricts the set of allowed field names
    and prevents typos, raising ``KeyError`` when referring to undefined fields.
    Additionally, fields can be used to define metadata and control the way
    data is processed internally. Please refer to the :ref:`documentation
    about fields <topics-items-fields>` for additional information.

    Unlike instances of :class:`dict`, instances of :class:`Item` may be
    :ref:`tracked <topics-leaks-trackrefs>` to debug memory leaks.
    """

    fields: Dict[str, Field]

    def __init__(self, *args, **kwargs):
        self._values = {}
        if args or kwargs:
            items = dict(*args, **kwargs)
            for key in items:
                if key in self.fields:
                    self[key] = items[key]
                else:
                    message = f"{self.__class__.__name__} does not support field: {key}"
                    raise KeyError(message)

    def __getitem__(self, key: str) -> any:
        return self._values[key]

    def __setitem__(self, key: str, value: any):
        if key in self.fields:
            self._values[key] = copy(value)
        else:
            message = f"{self.__class__.__name__} does not support field: {key}"
            raise KeyError(message)

    def __delitem__(self, key: str):
        del self._values[key]

    def __getattr__(self, name: str) -> any:
        if name in self.fields:
            message = f"Use item[{name!r}] to get field value"
            raise AttributeError(message)
        raise AttributeError(name)

    def __setattr__(self, name: str, value: any):
        if not name.startswith("_"):
            message = f"Use item[{name!r}] = {value!r} to set field value"
            raise AttributeError(message)
        super().__setattr__(name, value)

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self):
        return iter(self._values)

    def keys(self):
        return self._values.keys()

    def __repr__(self) -> str:
        return pformat(dict(self))

    def copy(self):
        return self.__class__(self)

    def deepcopy(self):
        """Return a :func:`~copy.deepcopy` of this item."""
        return deepcopy(self)
