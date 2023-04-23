from typing import Any, Callable, Deque, Dict, Iterable, List, Tuple, Union
from collections import defaultdict, deque
from scrapy import Spider
from scrapy.exceptions import NotConfigured
from scrapy.settings import Settings
from scrapy.utils.misc import create_instance, load_object
from scrapy.utils.defer import process_chain, process_parallel


class MiddlewareManager:
    """Base class for implementing middleware managers"""

    COMPONENT_NAME: str = "foo middleware"

    def __init__(self, *middlewares: Any) -> None:
        self.middlewares: List[Any] = middlewares
        self.methods: Dict[str, Deque[Union[None, Callable, Tuple[Callable, Callable]]]] = defaultdict(deque)
        for middleware in middlewares:
            self._add_middleware(middleware)

    @classmethod
    def _get_middleware_list_from_settings(cls, settings: Settings) -> List[str]:
        raise NotImplementedError

    @classmethod
    def from_settings(cls, settings: Settings, crawler: Any = None) -> "MiddlewareManager":
        middleware_list = cls._get_middleware_list_from_settings(settings)
        middlewares = []
        enabled_middlewares = []
        for class_path in middleware_list:
            try:
                middleware_class = load_object(class_path)
                middleware = create_instance(middleware_class, settings, crawler)
                middlewares.append(middleware)
                enabled_middlewares.append(class_path)
            except NotConfigured as exception:
                if exception.args:
                    class_name = class_path.split(".")[-1]
                    logger.warning(
                        f"Disabled {class_name}: {exception.args[0]}",
                        extra={"crawler": crawler},
                    )

        logger.info(
            f"Enabled {cls.COMPONENT_NAME}s:\n{pprint.pformat(enabled_middlewares)}",
            extra={"crawler": crawler},
        )
        return cls(*middlewares)

    @classmethod
    def from_crawler(cls, crawler: Any) -> "MiddlewareManager":
        return cls.from_settings(crawler.settings, crawler)

    def _add_middleware(self, middleware: Any) -> None:
        if hasattr(middleware, "open_spider"):
            self.methods["open_spider"].append(middleware.open_spider)
        if hasattr(middleware, "close_spider"):
            self.methods["close_spider"].appendleft(middleware.close_spider)

    def _process_parallel(self, method_name: str, obj: Any, *args: Any) -> Any:
        methods = self.methods[method_name]
        return process_parallel(cast(Iterable[Callable], methods), obj, *args)

    def _process_chain(self, method_name: str, obj: Any, *args: Any) -> Any:
        methods = self.methods[method_name]
        return process_chain(cast(Iterable[Callable], methods), obj, *args)

    def open_spider(self, spider: Spider) -> Any:
        return self._process_parallel("open_spider", spider)

    def close_spider(self, spider: Spider) -> Any:
        return self._process_parallel("close_spider", spider)
