import os
import signal
from typing import Any, Callable

from itemadapter import is_item
from twisted.internet import defer, threads
from twisted.python import threadable
from w3lib.url import any_to_uri

from scrapy.crawler import Crawler
from scrapy.exceptions import IgnoreRequest
from scrapy.http import Request, Response
from scrapy.settings import Settings
from scrapy.spiders import Spider
from scrapy.utils.conf import get_config
from scrapy.utils.console import DEFAULT_PYTHON_SHELLS, start_python_console
from scrapy.utils.datatypes import SequenceExclude
from scrapy.utils.misc import load_object
from scrapy.utils.reactor import is_asyncio_reactor_installed, set_asyncio_event_loop
from scrapy.utils.response import open_in_browser


DEFAULT_ITEM_CLASS = "scrapy.item.Item"


class Shell:
    RELEVANT_CLASSES = (Crawler, Spider, Request, Response, Settings)
    SHELL_OPTION = "settings:shell"

    def __init__(
        self,
        crawler: Crawler,
        update_vars: Callable[[dict[str, Any]], None] = None,
        code: str = None,
    ):
        self.crawler = crawler
        self.update_vars = update_vars or (lambda x: None)
        self.item_class = load_object(DEFAULT_ITEM_CLASS)
        self.spider = None
        self.in_thread = not threadable.isInIOThread()
        self.code = code
        self.vars: dict[str, Any] = {}

    def start(
        self,
        url: str = None,
        request: Request = None,
        response: Response = None,
        spider: Spider = None,
        redirect: bool = True,
    ) -> None:
        self._ignore_interrupt()
        if url:
            self._fetch(url, spider, redirect=redirect)
        elif request:
            self._fetch(request, spider)
        elif response:
            request = response.request
            self._populate_vars(response, request, spider)
        else:
            self._populate_vars()
        if self.code:
            print(eval(self.code, globals(), self.vars))
        else:
            self._start_python_console()

    def _ignore_interrupt(self) -> None:
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    def _fetch(
        self, request_or_url: str, spider: Spider = None, redirect: bool = True, **kwargs
    ) -> None:
        from twisted.internet import reactor

        if isinstance(request_or_url, Request):
            request = request_or_url
        else:
            url = any_to_uri(request_or_url)
            request = Request(url, dont_filter=True, **kwargs)
            if redirect:
                request.meta["handle_httpstatus_list"] = SequenceExclude(
                    range(300, 400)
                )
            else:
                request.meta["handle_httpstatus_all"] = True
        response = None
        try:
            response, spider = threads.blockingCallFromThread(
                reactor, self._schedule, request, spider
            )
        except IgnoreRequest:
            pass
        self._populate_vars(response, request, spider)

    def _populate_vars(
        self, response: Response = None, request: Request = None, spider: Spider = None
    ) -> None:
        import scrapy

        self.vars["scrapy"] = scrapy
        self.vars["crawler"] = self.crawler
        self.vars["item"] = self.item_class()
        self.vars["settings"] = self.crawler.settings
        self.vars["spider"] = spider
        self.vars["request"] = request
        self.vars["response"] = response
        if self.in_thread:
            self.vars["fetch"] = self._fetch
        self.vars["view"] = open_in_browser
        self.vars["shelp"] = self.print_help
        self.update_vars(self.vars)
        if not self.code:
            self.vars["banner"] = self._get_help()

    def _start_python_console(self) -> None:
        cfg = get_config()
        env = os.environ.get("SCRAPY_PYTHON_SHELL")
        shells = (
            env.strip().lower().split(",") if env else [cfg.get(Shell.SHELL_OPTION)]
        )
        shells += ["python"]
        start_python_console(
            self.vars, shells=DEFAULT_PYTHON_SHELLS.keys(), banner=self.vars.pop("banner", "")
        )

    def print_help(self) -> None:
        print(self._get_help())

    def _get_help(self) -> str:
        b = []
        b.append("Available Scrapy objects:")
        b.append(
            f"  scrapy     scrapy module (contains {', '.join(c.__name__ for c in Shell.RELEVANT_CLASSES)})"
        )
        for k, v in sorted(self.vars.items()):
            if self._is_relevant(v):
                b.append(f"  {k:<10} {v}")
        b.append("Useful shortcuts:")
        if self.in_thread:
            b.append(
                "  fetch(url[, redirect=True]) "
                "Fetch URL and update local objects (by default, redirects are followed)"
            )
            b.append(
                "  fetch(req)                  "
                "Fetch a scrapy.Request and update local objects "
            )
        b.append("  shelp()           Shell help (print this help)")
        b.append("  view(response)    View response in a browser")
        return "\n".join(f"[s] {line}" for line in b)

    def _is_relevant(self, value: Any) -> bool:
        return isinstance(value, self.RELEVANT_CLASSES) or is_item(value)

    def _schedule(self, request: Request, spider: Spider) -> defer.Deferred:
        if is_asyncio_reactor_installed():
            event_loop_path = self.crawler.settings["ASYNCIO_EVENT_LOOP"]
            set_asyncio_event_loop(event_loop_path)
        spider = self._open_spider(request, spider)
        d = _request_deferred(request)
        d.addCallback(lambda x: (x, spider))
        self.crawler.engine.crawl(request)
        return d

    def _open_spider(self, request: Request, spider: Spider) -> Spider:
        if self.spider:
            return self.spider
        if spider is None:
            spider = self.crawler.spider or self.crawler._create_spider()
        self.crawler.spider = spider
        self.crawler.engine.open_spider(spider, close_if_idle=False)
        self.spider = spider
        return spider


def inspect_response(response: Response, spider: Spider) -> None:
    sigint_handler = signal.getsignal(signal.SIGINT)
    Shell(spider.crawler).start(response=response, spider=spider)
    signal.signal(signal.SIGINT, sigint_handler)


def _request_deferred(request: Request) -> defer.Deferred:
    def _restore_callbacks(result) -> Any:
        request.callback = request_callback
        request.errback = request_errback
        return result

    request_callback = request.callback
    request_errback = request.errback
    d = defer.Deferred()
    d.addBoth(_restore_callbacks)
    if request.callback:
        d.addCallbacks(request.callback, request.errback)
    request.callback, request.errback = d.callback, d.errback
    return d