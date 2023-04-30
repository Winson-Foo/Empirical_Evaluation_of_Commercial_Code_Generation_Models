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
        self.dict: Optional[dict] = None

    def get_path(self, path: str) -> JSONTYPES:
        # To improve performance, don't use get_jsonpath if the key is easily accessible using dotted path syntax (e.g. "path.to.my.key")
        is_dotted_path = True if re.match("^([\w]+[.]?)+$", path) else False
        if is_dotted_path:
            return [self._get_dict_by_path(self.dict, path)]
        else:
            return get_jsonpath(self.dict, path)

    def set_path(self, path: str, value: JSONTYPES):
        # To improve performance, don't use set_jsonpath if the key is easily accessible using dotted path syntax (e.g. "path.to.my.key")
        is_dotted_path = True if re.match("^([\w]+[.]?)+$", path) else False
        if is_dotted_path:
            self._update_dict_by_path(self.dict, path, value)
        else:
            set_jsonpath(self.dict, path, value)

    def _update_dict_by_path(self, obj, path, value):
        """ 
        Update a nested dictionary using dotted path of a key and its value
        Example: _update_dict_by_path(my_dict, "path.to.my.key", 7)
        """
        keys = path.split('.')
        for key in keys[:-1]:
            obj = obj.setdefault(key, {})
        obj[keys[-1]] = value

    def _get_dict_by_path(self, obj, path):
        """ 
        Retrieve a value in a nested dictionary using dotted path of its key
        Example: _get_dict_by_path(my_dict, "path.to.my.key")
        """
        keys = path.split('.')
        for key in keys:
            try:
                obj = obj[key]
            except (KeyError, TypeError):
                return None
        return obj


class RequestDelegate(Delegate):
    def __init__(self, dict_request: dict,
                 uhost: str = defaults.uhost,
                 uport: int = defaults.uport,
                 ussl: bool = defaults.ussl,
                 query_delim: str = defaults.query_delim,
                 topn: int = defaults.topn,
                 query_prep: str = defaults.query_prep,
                 topk_path: str = defaults.topk_path,
                 default_topk: int = defaults.default_topk,
                 query_path: str = defaults.query_path,
                 rerank_cids: bool = defaults.rerank_cids,
                 choices_path: str = defaults.choices_path,
                 cvalues_path: str = defaults.cvalues_path,
                 cids_path: str = defaults.cids_path,
                 filter_results: bool = defaults.filter_results,
                 qa_threshold: float = defaults.qa_threshold):
        super().__init__()
        self.dict = dict_request
        self.uhost = str(uhost)
        self.uport = int(uport)
        self.ussl = bool(ussl)
        self.query_path = str(query_path)
        self.query_delim = str(query_delim)
        self.query_prep = str(query_prep)
        self.topn = int(topn)
        self.topk_path = str(topk_path)
        self.default_topk = int(default_topk)
        self.rerank_cids = bool(rerank_cids)
        self.choices_path = str(choices_path)
        self.cvalues_path = str(cvalues_path)
        self.cids_path = str(cids_path)
        self.filter_results = bool(filter_results)
        self.qa_threshold = float(qa_threshold)

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
    def __init__(self, dict_response: dict, request: RequestDelegate):
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
            f'{self.request.choices_path}.[*].{self.request.cids_path}'
        )

    @property
    def cvalues(self) -> list:
        return self.get_path(
            f'{self.request.choices_path}.[*].{self.request.cvalues_path}'
        )