import os
import signal
from typing import Any, Callable, Dict, Optional

from itemadapter import is_item
from twisted.internet import defer, threads, reactor
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
from scrapy.utils.reactor import (
    is_asyncio_reactor_installed,
    set_asyncio_event_loop,
)
from scrapy.utils.response import open_in_browser


class Shell:
    relevant_classes = (Crawler, Spider, Request, Response, Settings)

    def __init__(
        self,
        crawler: Crawler,
        update_vars: Optional[Callable[[Dict[str, Any]], None]] = None,
        code: Optional[str] = None,
    ) -> None:
        self.crawler = crawler
        self.update_vars = update_vars or (lambda x: None)
        self.item_class = load_object(crawler.settings["DEFAULT_ITEM_CLASS"])
        self.spider: Optional[Spider] = None
        self.in_thread = not threadable.isInIOThread()
        self.code = code
        self.vars: Dict[str, Any] = {}

    def start(
        self,
        url: Optional[str] = None,
        request: Optional[Request] = None,
        response: Optional[Response] = None,
        spider: Optional[Spider] = None,
        redirect: bool = True,
    ) -> None:
        # disable accidental Ctrl-C key press from shutting down the engine
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if url:
            self.fetch(url, spider, redirect=redirect)
        elif request:
            self.fetch(request, spider)
        elif response:
            request = response.request
            self.populate_vars(response, request, spider)
        else:
            self.populate_vars()
        if self.code:
            print(eval(self.code, globals(), self.vars))
        else:
            """
            Detect interactive shell setting in scrapy.cfg
            e.g.: ~/.config/scrapy.cfg or ~/.scrapy.cfg
            [settings]
            # shell can be one of ipython, bpython or python;
            # to be used as the interactive python console, if available.
            # (default is ipython, fallbacks in the order listed above)
            shell = python
            """
            cfg = get_config()
            section, option = "settings", "shell"
            env = os.environ.get("SCRAPY_PYTHON_SHELL")
            shells = []
            if env:
                shells += env.strip().lower().split(",")
            elif cfg.has_option(section, option):
                shells += [cfg.get(section, option).strip().lower()]
            else:  # try all by default
                shells += DEFAULT_PYTHON_SHELLS.keys()
            # always add standard shell as fallback
            shells += ["python"]
            start_python_console(
                self.vars, shells=shells, banner=self.vars.pop("banner", "")
            )

    def _schedule(self, request: Request, spider: Optional[Spider]) -> defer.Deferred:
        if is_asyncio_reactor_installed():
            # set the asyncio event loop for the current thread
            event_loop_path = self.crawler.settings["ASYNCIO_EVENT_LOOP"]
            set_asyncio_event_loop(event_loop_path)
        spider = self._open_spider(request, spider)
        d = _request_deferred(request)
        d.addCallback(lambda x: (x, spider))
        self.crawler.engine.crawl(request)
        return d

    def _open_spider(self, request: Request, spider: Optional[Spider]) -> Spider:
        if self.spider:
            return self.spider

        if spider is None:
            spider = self.crawler.spider or self.crawler._create_spider()

        self.crawler.spider = spider
        self.crawler.engine.open_spider(spider, close_if_idle=False)
        self.spider = spider
        return spider

    def fetch(
        self,
        request_or_url: Any,
        spider: Optional[Spider] = None,
        redirect: bool = True,
        **kwargs: Any,
    ) -> None:

        if isinstance(request_or_url, Request):
            request = request_or_url
        else:
            url = any_to_uri(request_or_url)
            request = Request(url, dont_filter=True, **kwargs)
            if redirect:
                request.meta["handle_httpstatus_list"] = SequenceExclude(range(300, 400))
            else:
                request.meta["handle_httpstatus_all"] = True
        response = None
        try:
            response, spider = threads.blockingCallFromThread(
                reactor, self._schedule, request, spider
            )
        except IgnoreRequest:
            pass
        self.populate_vars(response, request, spider)

    def populate_vars(
        self, response: Optional[Response] = None, request: Optional[Request] = None, spider: Optional[Spider] = None,
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
            self.vars["fetch"] = self.fetch
        self.vars["view"] = open_in_browser
        self.vars["shelp"] = self.print_help
        self.update_vars(self.vars)
        if not self.code:
            self.vars["banner"] = self.get_help()

    def print_help(self) -> None:
        print(self.get_help())

    def get_help(self) -> str:
        b = []
        b.append("Available Scrapy objects:")
        b.append("  scrapy     scrapy module (contains scrapy.Request, scrapy.Selector, etc)")
        for k, v in sorted(self.vars.items()):
            if self._is_relevant(v):
                b.append(f"  {k:<10} {v}")
        b.append("Useful shortcuts:")
        if self.in_thread:
            b.append(
                "  fetch(url[, redirect=True]) Fetch URL and update local objects (by default, redirects are followed)"
            )
            b.append(
                "  fetch(req)                  Fetch a scrapy.Request and update local objects "
            )
        b.append("  shelp()           Shell help (print this help)")
        b.append("  view(response)    View response in a browser")

        return "\n".join(f"[s] {line}" for line in b)

    def _is_relevant(self, value: Any) -> bool:
        return isinstance(value, self.relevant_classes) or is_item(value)


def inspect_response(response: Response, spider: Spider) -> None:
    """Open a shell to inspect the given response"""

    sigint_handler = signal.getsignal(signal.SIGINT)
    Shell(spider.crawler).start(response=response, spider=spider)
    signal.signal(signal.SIGINT, sigint_handler)


def _request_deferred(request: Request) -> defer.Deferred:
    """Wrap a request inside a Deferred.

    This function is harmful, do not use it until you know what you are doing.

    This returns a Deferred whose first pair of callbacks are the request
    callback and errback. The Deferred also triggers when the request
    callback/errback is executed (i.e. when the request is downloaded)

    WARNING: Do not call request.replace() until after the deferred is called.
    """
    request_callback = request.callback
    request_errback = request.errback

    def _restore_callbacks(result: Any) -> Any:
        request.callback = request_callback
        request.errback = request_errback
        return result

    d = defer.Deferred()
    d.addBoth(_restore_callbacks)
    if request.callback:
        d.addCallbacks(request.callback, request.errback)

    request.callback, request.errback = d.callback, d.errback
    return d 