# delegate.py
import re
from typing import Optional, Union, List

from nboost import defaults
from nboost.helpers import get_jsonpath, set_jsonpath, JSONTYPES, flatten
from nboost.exceptions import MissingQuery, InvalidChoices


class Delegate:
    def __init__(self) -> None:
        self.data: Optional[dict] = None

    def get_path(self, path: str) -> Union[JSONTYPES, List[JSONTYPES]]:
        is_dotted_path = True if re.match("^([\w]+[.]?)+$", path) else False
        if is_dotted_path:
            return [self._get_dict_by_path(self.data, path)]
        else:
            return get_jsonpath(self.data, path)

    def set_path(self, path: str, value: JSONTYPES) -> None:
        is_dotted_path = True if re.match("^([\w]+[.]?)+$", path) else False
        if is_dotted_path:
            self._update_dict_by_path(self.data, path, value)
        else:
            set_jsonpath(self.data, path, value)

    def _update_dict_by_path(self, obj: dict, path: str, value: JSONTYPES) -> None:
        split_path = path.split('.', maxsplit=1)

        if len(split_path) == 1:
            obj[split_path[0]] = value
        else:
            if split_path[0] not in obj.keys():
                obj[split_path[0]] = {}
            self._update_dict_by_path(obj[split_path[0]], split_path[1], value)

    def _get_dict_by_path(self, obj: dict, path: str) -> JSONTYPES:
        split_path = path.split('.', maxsplit=1)

        if len(split_path) == 1:
            return obj[split_path[0]]
        else:
            return self._get_dict_by_path(obj[split_path[0]], split_path[1])


class RequestDelegate(Delegate):
    def __init__(self, data: dict, **kwargs) -> None:
        super().__init__()
        self.data = data
        self.uhost = str(kwargs.get('uhost', defaults.uhost))
        self.uport = int(kwargs.get('uport', defaults.uport))
        self.ussl = bool(kwargs.get('ussl', defaults.ussl))
        self.query_path = str(kwargs.get('query_path', defaults.query_path))
        self.query_delim = str(kwargs.get('query_delim', defaults.query_delim))
        self.query_prep = str(kwargs.get('query_prep', defaults.query_prep))
        self.topn = int(kwargs.get('topn', defaults.topn))
        self.topk_path = str(kwargs.get('topk_path', defaults.topk_path))
        self.default_topk = int(kwargs.get('default_topk', defaults.default_topk))
        self.rerank_cids = bool(kwargs.get('rerank_cids', defaults.rerank_cids))
        self.choices_path = str(kwargs.get('choices_path', defaults.choices_path))
        self.cvalues_path = str(kwargs.get('cvalues_path', defaults.cvalues_path))
        self.cids_path = str(kwargs.get('cids_path', defaults.cids_path))
        self.filter_results = bool(kwargs.get('filter_results', defaults.filter_results))
        self.qa_threshold = float(kwargs.get('qa_threshold', defaults.qa_threshold))

    @property
    def topk(self) -> int:
        topks = self.get_path(self.topk_path)
        return int(topks[0]) if topks else self.default_topk

    @topk.setter
    def topk(self, value: int) -> None:
        self.set_path(self.topk_path, value)

    @property
    def query(self) -> str:
        queries = self.get_path(self.query_path)
        query = self.query_delim.join(queries)

        if not query:
            raise MissingQuery

        return eval(self.query_prep)(query)


class ResponseDelegate(Delegate):
    def __init__(self, data: dict, request: RequestDelegate, **kwargs) -> None:
        super().__init__()
        self.data = data
        self.request = request

    @property
    def choices(self) -> List[str]:
        choices = self.get_path(self.request.choices_path)

        if not isinstance(choices, list):
            raise InvalidChoices('choices were not a list')

        return flatten(choices)

    @choices.setter
    def choices(self, value: List[str]) -> None:
        self.set_path(self.request.choices_path, value)

    @property
    def cids(self) -> List[Union[str, int]]:
        return self.get_path(f'{self.request.choices_path}.[*].{self.request.cids_path}')

    @property
    def cvalues(self) -> List[str]:
        return self.get_path(f'{self.request.choices_path}.[*].{self.request.cvalues_path}')