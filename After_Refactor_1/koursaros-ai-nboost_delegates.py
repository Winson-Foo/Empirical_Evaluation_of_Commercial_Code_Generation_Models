import re
from typing import Optional
from nboost.helpers import get_jsonpath, set_jsonpath, JSONTYPES, flatten
from nboost.exceptions import *
from nboost import defaults


class Delegate:
    """A Class that parses the attributes of a request or response. It is
     configured by command line and runtime arguments. Also used for setting
     the request/response prior to preparation."""
    def __init__(self):
        self.dict = None  # type: Optional[dict]

    def is_dotted_path(self, path: str) -> bool:
        """Check if the path is in dotted path syntax."""
        return bool(re.match("^([\w]+[.]?)+$", path))

    def get_path(self, path: str) -> JSONTYPES:
        """Get the value for the given path."""
        if self.is_dotted_path(path):
            return [self._get_dict_by_path(self.dict, path)]
        else:
            return get_jsonpath(self.dict, path)

    def set_path(self, path: str, value: JSONTYPES):
        """Set the value for the given path."""
        if self.is_dotted_path(path):
            self._update_dict_by_path(self.dict, path, value)
        else:
            set_jsonpath(self.dict, path, value)

    def _update_dict_by_path(self, obj, path, value):
        """ 
        Update a nested dictionary using dotted path of a key and its value
        Example: _update_dict_by_path(my_dict, "path.to.my.key", 7)
        """
        split_path = path.split('.', maxsplit=1)

        if len(split_path) == 1:
            obj[split_path[0]] = value
        else:
            if split_path[0] not in obj.keys():
                obj[split_path[0]] = {}
            self._update_dict_by_path(obj[split_path[0]], split_path[1], value)

    def _get_dict_by_path(self, obj, path):
        """ 
        Retrieve a value in a nested dictionary using dotted path of its key
        Example: _get_dict_by_path(my_dict, "path.to.my.key")
        """
        split_path = path.split('.', maxsplit=1)

        if len(split_path) == 1:
            return obj[split_path[0]]
        else:
            return self._get_dict_by_path(obj[split_path[0]], split_path[1])


class RequestDelegate(Delegate):
    def __init__(self, dict_request: dict, **kwargs):
        super().__init__()
        self.uhost = getattr(defaults, "uhost", "localhost")
        self.uport = getattr(defaults, "uport", 80)
        self.ussl = getattr(defaults, "ussl", False)
        self.query_delim = getattr(defaults, "query_delim", " ")
        self.topn = getattr(defaults, "topn", 10)
        self.query_prep = getattr(defaults, "query_prep", lambda x: x)
        self.topk_path = getattr(defaults, "topk_path", "top_k")
        self.default_topk = getattr(defaults, "default_topk", 3)
        self.query_path = getattr(defaults, "query_path", "query")
        self.rerank_cids = getattr(defaults, "rerank_cids", False)
        self.choices_path = getattr(defaults, "choices_path", "choices")
        self.cvalues_path = getattr(defaults, "cvalues_path", "cvalues")
        self.cids_path = getattr(defaults, "cids_path", "cids")
        self.filter_results = getattr(defaults, "filter_results", False)
        self.qa_threshold = getattr(defaults, "qa_threshold", 0.5)

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.dict = dict_request

    @property
    def topk(self) -> int:
        topks = self.get_path(self.topk_path)
        return int(topks[0]) if topks else self.default_topk

    @topk.setter
    def topk(self, value: int):
        self.set_path(self.topk_path, value)

    @property
    def query(self) -> str:
        queries = self.get_path(self.query_path)
        query = self.query_delim.join(queries)

        # check for errors
        if not query:
            raise MissingQuery

        return eval(self.query_prep)(query)


class ResponseDelegate(Delegate):
    def __init__(self, dict_response: dict, request: RequestDelegate, **_):
        super().__init__()
        self.dict = dict_response
        self.request = request

    @property
    def choices(self) -> list:
        choices = self.get_path(self.request.choices_path)

        if not isinstance(choices, list):
            raise InvalidChoices('choices were not a list')

        return flatten(choices)

    @choices.setter
    def choices(self, value: list):
        self.set_path(self.request.choices_path, value)

    @property
    def cids(self) -> list:
        return self.get_path(
            self.request.choices_path + '.[*].' + self.request.cids_path
        )

    @property
    def cvalues(self) -> list:
        return self.get_path(
            self.request.choices_path + '.[*].' + self.request.cvalues_path
        )
