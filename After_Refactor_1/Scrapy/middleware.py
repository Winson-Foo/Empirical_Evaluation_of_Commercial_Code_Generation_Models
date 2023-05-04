import pprint
import logging
from typing import Any, Callable, Deque, Dict, List, Iterable, Tuple, Union
from collections import deque, defaultdict

from twisted.internet.defer import Deferred

from scrapy import Spider
from scrapy.exceptions import NotConfigured
from scrapy.settings import Settings
from scrapy.utils.defer import process_chain, process_parallel
from scrapy.utils.misc import create_instance, load_object


logger = logging.getLogger(__name__)


class Middleware:
    """Base class for implementing middleware"""

    def open_spider(self, spider: Spider) -> Deferred:
        """
        Called when spider is opened.
        """

    def close_spider(self, spider: Spider) -> Deferred:
        """
        Called when spider is closed.
        """

    def process_spider_input(
            self, response: Any, spider: Spider) -> Tuple[Any, Union[Deferred, None]]:
        """
        Process spider input
        """

    def process_spider_output(
            self, response: Any, result: Union[List, Tuple], spider: Spider
    ) -> Iterable[Union[Tuple, Dict]]:
        """
        Process spider output
        """


class MiddlewareManager:
    """Middleware Manager"""

    def __init__(self, middlewares: List[Middleware]) -> None:
        self.middlewares = middlewares
        self.methods: Dict[
            str, Deque[Union[None, Callable, Tuple[Callable, Callable]]]
        ] = defaultdict(deque)
        self._add_middlewares()

    def _add_middlewares(self) -> None:
        for middleware in self.middlewares:
            if hasattr(middleware, "open_spider"):
                self.methods["open_spider"].append(middleware.open_spider)
            if hasattr(middleware, "close_spider"):
                self.methods["close_spider"].appendleft(middleware.close_spider)

    def open_spider(self, spider: Spider) -> Deferred:
        methods = self.methods["open_spider"]
        return process_parallel(methods, spider)

    def close_spider(self, spider: Spider) -> Deferred:
        methods = self.methods["close_spider"]
        return process_parallel(methods, spider)

    def process_spider_input(
            self, response: Any, spider: Spider) -> Tuple[Any, Union[Deferred, None]]:
        for middleware in self.middlewares:
            result = middleware.process_spider_input(response, spider)
            if result:
                response, deferred = result
                if deferred:
                    return response, deferred
        return response, None

    def process_spider_output(
            self, response: Any, result: Union[List, Tuple], spider: Spider
    ) -> Iterable[Union[Tuple, Dict]]:
        for middleware in self.middlewares:
            result = middleware.process_spider_output(response, result, spider)
            if not result:
                break
            response, result = result
            if isinstance(result, Deferred):
                return self._process_deferred_result(response, result, spider)
        yield from result

    def _process_deferred_result(
            self, response: Any, deferred: Deferred, spider: Spider
    ) -> Iterable[Union[Tuple, Dict]]:
        def process_result(result):
            response, result = result
            yield from self.process_spider_output(response, result, spider)

        deferred.addCallback(process_result)
        return deferred.result


class MiddlewareSettingsManager:
    """Middleware Settings Manager"""

    COMPONENT_NAME = "middleware"

    @classmethod
    def from_crawler(cls, crawler: Spider) -> MiddlewareManager:
        settings = crawler.settings
        mwlist = cls._get_middleware_list_from_settings(settings)
        enabled: List[Middleware] = []
        disabled = []
        for clspath in mwlist:
            try:
                middleware_class = load_object(clspath)
                middleware = create_instance(middleware_class, settings, crawler)
                enabled.append(middleware)
            except NotConfigured as e:
                if e.args:
                    clsname = clspath.split(".")[-1]
                    logger.warning(
                        "Disabled %(clsname)s: %(eargs)s",
                        {"clsname": clsname, "eargs": e.args[0]},
                        extra={"crawler": crawler},
                    )
                disabled.append(clspath)

        logger.info(
            "Enabled %(COMPONENT_NAME)ss:\n%(enabledlist)s",
            {
                "COMPONENT_NAME": cls.COMPONENT_NAME,
                "enabledlist": pprint.pformat(enabled),
            },
            extra={"crawler": crawler},
        )
        return MiddlewareManager(enabled)

    @classmethod
    def _get_middleware_list_from_settings(cls, settings: Settings) -> List[str]:
        raise NotImplementedError 