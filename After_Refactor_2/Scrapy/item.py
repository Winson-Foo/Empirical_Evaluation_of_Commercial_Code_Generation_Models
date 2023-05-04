from abc import ABCMeta
from collections.abc import MutableMapping
from copy import deepcopy
from pprint import pformat
from typing import Dict


class Field(dict):
    """Container of field metadata."""

class ItemMeta(ABCMeta):
    """
    Metaclass of Item that handles field definitions.
    """

    def __new__(mcs, class_name: str, bases, attrs: dict) -> object:
        classcell = attrs.pop("__classcell__", None)
        new_bases = tuple(base._class for base in bases if hasattr(base, "_class"))
        _class = super().__new__(mcs, "x_" + class_name, new_bases, attrs)

        fields = getattr(_class, "fields", {})
        new_attrs = {}
        for n in dir(_class):
            v = getattr(_class, n)
            if isinstance(v, Field):
                fields[n] = v
            elif n in attrs:
                new_attrs[n] = attrs[n]

        new_attrs["fields"] = fields
        new_attrs["_class"] = _class
        if classcell is not None:
            new_attrs["__classcell__"] = classcell
        return super().__new__(mcs, class_name, bases, new_attrs)


class Item(MutableMapping, object_ref, metaclass=ItemMeta):
    """
    Base class for scraped items.

    In Scrapy, an object is considered an item if it is an instance of either :class:`Item` or :class:`dict`, or any subclass. When the output of a
    spider callback is evaluated, only instances of :class:`Item` or
    :class:`dict` are passed to item pipelines.

    If you need instances of a custom class to be considered items by Scrapy,
    you must inherit from :class:`Item`.

    Items must declare :class:`Field` attributes, which are processed and stored
    in the `fields` attribute. This restricts the set of allowed field names
    and prevents typos, raising `KeyError` when referring to undefined fields.
    Additionally, fields can be used to define metadata and control the way
    data is processed internally. Please refer to the documentation
    about fields for additional information.

    Unlike instances of a dictionary, instances of :class:`Item` may be
    tracked to debug memory leaks.

    """

    fields: Dict[str, Field]

    def __init__(self, *args, **kwargs):
        """
        Initialize item instance.

        Parameters:
        -----------
        *args: tuple
            Position arguments.
        **kwargs: dict
            Keyword arguments.

        """
        self._values = {}
        if args or kwargs:
            for k, v in dict(*args, **kwargs).items():
                self[k] = v

    def __getitem__(self, key: str):
        """
        Return the field value associated with the given key.

        Parameters:
        -----------
        key: str
            The key to retrieve the value for.

        Returns:
        --------
        object
            The value associated with the key.

        """
        return self._values[key]

    def __setitem__(self, key: str, value: object):
        """
        Set the given key to the given value.

        Parameters:
        -----------
        key: str
            The key for the field.
        value: object
            The value to set for the key.

        Raises:
        -------
        KeyError
            If the key is not defined in the fields.

        """
        if key in self.fields:
            self._values[key] = value
        else:
            raise KeyError(f"{self.__class__.__name__} does not support field: {key}")

    def __delitem__(self, key: str):
        """
        Remove the field value associated with the given key.

        Parameters:
        -----------
        key : str
            The key of the value to remove.

        """
        del self._values[key]

    def __getattr__(self, name: str):
        """
        Get the attribute associated with the given name.

        Parameters:
        -----------
        name: str
            The name of the attribute to get.

        Raises:
        ------
        AttributeError
            If the attribute is not found.

        """
        if name in self.fields:
            raise AttributeError(f"Use item[{name!r}] to get field value")
        raise AttributeError(name)

    def __setattr__(self, name: str, value: object):
        """
        Set the attribute with the given name and value.

        Parameters:
        -----------
        name: str
            The name of the attribute to set.
        value: object
            The value to set for the attribute.

        Raises:
        -------
        AttributeError
            If the attribute is not set.

        """
        if not name.startswith("_"):
            raise AttributeError(f"Use item[{name!r}] = {value!r} to set field value")
        super().__setattr__(name, value)

    def __len__(self) -> int:
        """
        Return the number of defined fields in the item.

        Returns:
        --------
        int
            The number of defined fields

        """
        return len(self._values)

    def __iter__(self):
        """
        Return an iterator over the defined fields.

        Returns:
        --------
        iterator
            An iterator over the defined fields.

        """
        return iter(self._values)

    __hash__ = object_ref.__hash__

    def keys(self):
        """
        Return a new view of the items keys.

        Returns:
        --------
        view
            A new view of the items keys.

        """
        return self._values.keys()

    def __repr__(self):
        """
        Return formatted string representation of the item.

        Returns:
        --------
        str
            Formatted string representation of the item.

        """
        return pformat(dict(self))

    def copy(self):
        """
        Return a shallow copy of the item.

        Returns:
        --------
        Item
            A shallow copy of the item.

        """
        return self.__class__(self)

    def deepcopy(self):
        """
        Return a deep copy of the item.

        Returns:
        --------
        Item
            A deep copy of the item.

        """
        return deepcopy(self) 