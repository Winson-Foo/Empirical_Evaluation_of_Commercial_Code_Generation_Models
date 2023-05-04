from abc import ABCMeta
from collections.abc import MutableMapping
from copy import deepcopy
from typing import Dict


class Field(dict):
    pass


class ItemMeta(ABCMeta):
    def __new__(cls, name, bases, attrs):
        class_cell = attrs.pop("__classcell__", None)
        new_bases = tuple(base._class for base in bases if hasattr(base, "_class"))
        new_attrs = {}

        for attr_name, attr_value in attrs.items():
            if isinstance(attr_value, Field):
                new_attrs["fields"] = new_attrs.get("fields", {})
                new_attrs["fields"][attr_name] = attr_value
            else:
                new_attrs[attr_name] = attr_value

        new_attrs["_class"] = super().__new__(cls, name, new_bases, new_attrs)
        new_attrs["fields"] = getattr(new_attrs["_class"], "fields", {})
        return new_attrs["_class"]


class Item(MutableMapping, metaclass=ItemMeta):
    fields: Dict[str, Field]

    def __init__(self, *args, **kwargs):
        self._values = {}
        if args:
            if len(args) > 1:
                raise TypeError("__init__() takes at most 1 positional argument")
            self._values.update(dict(args[0]))

        self._values.update(kwargs)

    def __getitem__(self, key):
        return self._values[key]

    def __setitem__(self, key, value):
        if key not in self.fields:
            raise KeyError(f"{self.__class__.__name__} does not support field: {key}")
        self._values[key] = value

    def __delitem__(self, key):
        del self._values[key]

    def __getattr__(self, name):
        if name in self._values:
            return self._values[name]
        elif name in self.fields:
            raise AttributeError(f"Use item[{name!r}] to get field value")
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if name.startswith("_"):
            super().__setattr__(name, value)
        elif name in self._values:
            self._values[name] = value
        elif name in self.fields:
            self._values[name] = value
        else:
            raise AttributeError(f"{self.__class__.__name__} does not support field: {name}")

    def __len__(self):
        return len(self._values)

    def __iter__(self):
        return iter(self._values)

    def keys(self):
        return self._values.keys()

    def __repr__(self):
        return repr(dict(self))

    def copy(self):
        return self.__class__(self._values.copy())

    def deepcopy(self):
        return deepcopy(self) 