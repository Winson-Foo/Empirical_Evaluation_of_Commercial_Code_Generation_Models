import logging
import pprint
from collections import defaultdict, deque
from typing import Any, Callable, Deque, Dict, Iterable, List, Tuple, Union

from twisted.internet.defer import Deferred

from scrapy import Spider
from scrapy.exceptions import NotConfigured
from scrapy.settings import Settings
from scrapy.utils.misc import load_object, create_instance

logger = logging.getLogger(__name__)


class MiddlewareManager:
    """
    Base class for implementing middleware managers
    """

    component_name = "Spider middleware"

    def __init__(self, *middlewares: Any) -> None:
        self._middlewares: List[Any] = list(middlewares)
        # Only process_spider_output and process_spider_exception can be None.
        # Only process_spider_output can be a tuple, and only until _async compatibility methods are removed.
        self._methods: Dict[
            str, Deque[Union[None, Callable, Tuple[Callable, Callable]]]
        ] = defaultdict(deque)
        
        for middleware in self._middlewares:
            self._add_middleware(middleware)

    @classmethod
    def _get_middleware_list_from_settings(cls, settings: Settings) -> List:
        raise NotImplementedError

    @classmethod
    def from_settings(cls, settings: Settings, crawler=None):
        middleware_list = cls._get_middleware_list_from_settings(settings)
        middlewares = []
        enabled = []
        
        for clspath in middleware_list:
            try:
                middleware_cls = load_object(clspath)
                middleware = create_instance(middleware_cls, settings, crawler)
                middlewares.append(middleware)
                enabled.append(clspath)
            except NotConfigured as e:
                if e.args:
                    class_name = clspath.split(".")[-1]
                    logger.warning(
                        "Disabled %(class_name)s: %(error_message)s",
                        {"class_name": class_name, "error_message": e.args[0]},
                        extra={"crawler": crawler},
                    )

        logger.info(
            "Enabled %(component_name_plural)ss:\n%(enabled_list)s",
            {
                "component_name_plural": cls.component_name,
                "enabled_list": pprint.pformat(enabled),
            },
            extra={"crawler": crawler},
        )
        
        return cls(*middlewares)

    @classmethod
    def from_crawler(cls, crawler):
        return cls.from_settings(crawler.settings, crawler)

    def _add_middleware(self, middleware: Any) -> None:
        if hasattr(middleware, "process_request"):
            self._methods["process_request"].append(middleware.process_request)
        if hasattr(middleware, "process_spider_input"):
            self._methods["process_spider_input"].append(middleware.process_spider_input)
        if hasattr(middleware, "process_spider_output"):
            self._methods["process_spider_output"].append(middleware.process_spider_output)
        if hasattr(middleware, "process_spider_exception"):
            self._methods["process_spider_exception"].append(middleware.process_spider_exception)
        if hasattr(middleware, "process_start_requests"):
            self._methods["process_start_requests"].append(middleware.process_start_requests)
        if hasattr(middleware, "spider_opened"):
            self._methods["spider_opened"].append(middleware.spider_opened)
        if hasattr(middleware, "spider_closed"):
            self._methods["spider_closed"].appendleft(middleware.spider_closed)

    def _process_methods(self, method_name: str, obj: Any, *args: Any) -> Deferred:
        methods = self._methods[method_name]
        
        if method_name in ["process_spider_output", "process_spider_exception"]:
            return process_parallel(methods, obj, *args)
        
        return process_chain(methods, obj, *args)

    def process_request(self, request: Any, spider: Spider) -> Deferred:
        return self._process_methods("process_request", request, spider)

    def process_spider_input(self, response: Any, spider: Spider) -> Deferred:
        return self._process_methods("process_spider_input", response, spider)

    def process_spider_output(self, response: Any, result: Iterable, spider: Spider) -> Deferred:
        return self._process_methods("process_spider_output", response, result, spider)

    def process_spider_exception(self, response: Any, exception: Exception, spider: Spider) -> Deferred:
        return self._process_methods("process_spider_exception", response, exception, spider)

    def process_start_requests(self, start_requests: List, spider: Spider) -> List:
        methods = self._methods["process_start_requests"]
        return process_chain(methods, start_requests, spider)

    def spider_opened(self, spider: Spider) -> Deferred:
        return self._process_methods("spider_opened", spider)

    def spider_closed(self, spider: Spider) -> Deferred:
        return self._process_methods("spider_closed", spider) 