import logging
import pprint
from collections import deque, defaultdict
from typing import Any, Callable, Deque, Dict, Iterable, List, Tuple, Union

from scrapy import Spider
from scrapy.exceptions import NotConfigured
from scrapy.settings import Settings
from scrapy.utils.defer import process_chain, process_parallel
from scrapy.utils.misc import create_instance, load_object

logger = logging.getLogger(__name__)


class MiddlewareManager:
    """Base class for implementing middleware managers."""

    component_name = "foo middleware"

    def __init__(self, *middlewares: Any) -> None:
        self.middlewares: List[Any] = middlewares
        self.methods: Dict[str, Deque[Union[None, Callable, Tuple[Callable, Callable]]]] = defaultdict(deque)
        for middleware in middlewares:
            self._add_middleware(middleware)

    @classmethod
    def _get_mwlist_from_settings(cls, settings: Settings) -> List[str]:
        return settings.getlist('MIDDLEWARES')

    @classmethod
    def from_settings(cls, settings: Settings, crawler=None) -> 'MiddlewareManager':
        middlewares = []
        enabled = []
        for clspath in cls._get_mwlist_from_settings(settings):
            try:
                mwcls = load_object(clspath)
                mw = create_instance(mwcls, settings, crawler)
                middlewares.append(mw)
                enabled.append(clspath)
            except NotConfigured as e:
                clsname = clspath.split(".")[-1]
                logger.warning(
                    "Disabled %(clsname)s: %(eargs)s",
                    {"clsname": clsname, "eargs": e.args[0]},
                    extra={"crawler": crawler},
                )

        logger.info(
            "Enabled %(componentname)ss:\n%(enabledlist)s",
            {
                "componentname": cls.component_name,
                "enabledlist": pprint.pformat(enabled),
            },
            extra={"crawler": crawler},
        )
        return cls(*middlewares)

    @classmethod
    def from_crawler(cls, crawler) -> 'MiddlewareManager':
        return cls.from_settings(crawler.settings, crawler)

    def _add_middleware(self, middleware: Any) -> None:
        for methodname in ["open_spider", "close_spider"]:
            method = getattr(middleware, methodname, None)
            if method is not None:
                if methodname == "close_spider":
                    self.methods[methodname].appendleft(method)
                else:
                    self.methods[methodname].append(method)

    def _process_parallel(self, methodname: str, obj: Any, *args: Any) -> Deferred:
        return process_parallel(self.methods[methodname], obj, *args)

    def _process_chain(self, methodname: str, obj: Any, *args: Any) -> Deferred:
        return process_chain(self.methods[methodname], obj, *args)

    def open_spider(self, spider: Spider) -> Deferred:
        return self._process_parallel("open_spider", spider)

    def close_spider(self, spider: Spider) -> Deferred:
        return self._process_parallel("close_spider", spider) 