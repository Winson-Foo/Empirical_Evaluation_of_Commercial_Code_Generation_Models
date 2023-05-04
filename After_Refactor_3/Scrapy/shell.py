import os
import signal
import scrapy

from itemadapter import is_item
from twisted.internet import defer, threads
from twisted.python import threadable
from w3lib.url import any_to_uri

from scrapy.crawler import Crawler
from scrapy.exceptions import IgnoreRequest
from scrapy.http import Request, Response
from scrapy.settings import Settings
from scrapy.spiders import Spider
from scrapy.utils.console import DEFAULT_PYTHON_SHELLS, start_python_console
from scrapy.utils.datatypes import SequenceExclude
from scrapy.utils.misc import load_object
from scrapy.utils.reactor import is_asyncio_reactor_installed, set_asyncio_event_loop
from scrapy.utils.response import open_in_browser


class Shell:

    RELEVANT_CLASSES = (Crawler, Spider, Request, Response, Settings)

    def __init__(self, crawler, update_vars=None, code=None):
        self.crawler = crawler
        self.update_vars = update_vars or (lambda x: None)
        self.item_class = load_object(crawler.settings["DEFAULT_ITEM_CLASS"])
        self.spider = None
        self.inthread = not threadable.isInIOThread()
        self.code = code
        self.vars = {}

    def start(self, url=None, request=None, response=None, spider=None, redirect=True):
        self._disable_interrupt_signal()
        if url:
            self._fetch_url(url, spider, redirect=redirect)
        elif request:
            self._fetch_request(request, spider)
        elif response:
            self._populate_vars_from_response(response, request, spider)
        else:
            self._populate_vars()
            if self.code:
                print(eval(self.code, globals(), self.vars))
            else:
                self._start_console()

    def _disable_interrupt_signal(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    def _fetch_url(self, url, spider, redirect):
        request = Request(url, dont_filter=True)
        if redirect:
            request.meta["handle_httpstatus_list"] = SequenceExclude(range(300, 400))
        else:
            request.meta["handle_httpstatus_all"] = True
        self._schedule(request, spider)

    def _fetch_request(self, request, spider):
        self._schedule(request, spider)

    def _populate_vars_from_response(self, response, request, spider):
        self._populate_vars()
        self.vars["response"] = response
        self.vars["request"] = request
        self.vars["spider"] = spider

    def _populate_vars(self):
        self.vars["scrapy"] = scrapy
        self.vars["crawler"] = self.crawler
        self.vars["item"] = self.item_class()
        self.vars["settings"] = self.crawler.settings
        self.vars["view"] = open_in_browser
        self.vars["shelp"] = self.print_help
        self.update_vars(self.vars)

    def _start_console(self):
        shells = self._get_shell_options()
        self.vars["banner"] = self._get_help()
        start_python_console(self.vars, shells=shells, banner=self.vars.pop("banner", ""))

    def _get_shell_options(self):
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
        return shells

    def print_help(self):
        print(self._get_help())

    def _get_help(self):
        relevant_vars = [f"  {k:<10} {v}" for k, v in self.vars.items() if self._is_relevant(v)]
        return f"Available Scrapy objects:\n{'\n'.join(relevant_vars)}\nUseful shortcuts:\n  fetch(url[, redirect=True]) Fetch URL and update local objects (by default, redirects are followed)\n  fetch(req)                  Fetch a scrapy.Request and update local objects\n  shelp()           Shell help (print this help)\n  view(response)    View response in a browser"

    def _is_relevant(self, value):
        return isinstance(value, self.RELEVANT_CLASSES) or is_item(value)

    def _schedule(self, request, spider):
        if is_asyncio_reactor_installed():
            # set the asyncio event loop for the current thread
            event_loop_path = self.crawler.settings["ASYNCIO_EVENT_LOOP"]
            set_asyncio_event_loop(event_loop_path)

        spider = self._open_spider(request, spider)
        d = _request_deferred(request)
        d.addCallback(lambda x: (x, spider))
        self.crawler.engine.crawl(request)
        return d

    def _open_spider(self, request, spider):
        if self.spider:
            return self.spider

        if spider is None:
            spider = self.crawler.spider or self.crawler._create_spider()

        self.crawler.spider = spider
        self.crawler.engine.open_spider(spider, close_if_idle=False)
        self.spider = spider
        return spider


def inspect_response(response, spider):
    """Open a shell to inspect the given response"""
    # Shell.start removes the SIGINT handler, so save it and re-add it after
    # the shell has closed
    sigint_handler = signal.getsignal(signal.SIGINT)
    Shell(spider.crawler).start(response=response, spider=spider)
    signal.signal(signal.SIGINT, sigint_handler)


def _request_deferred(request):
    """Wrap a request inside a Deferred.

    This function is harmful, do not use it until you know what you are doing.

    This returns a Deferred whose first pair of callbacks are the request
    callback and errback. The Deferred also triggers when the request
    callback/errback is executed (i.e. when the request is downloaded)

    WARNING: Do not call request.replace() until after the deferred is called.
    """
    request_callback = request.callback
    request_errback = request.errback

    def _restore_callbacks(result):
        request.callback = request_callback
        request.errback = request_errback
        return result

    d = defer.Deferred()
    d.addBoth(_restore_callbacks)
    if request.callback:
        d.addCallbacks(request.callback, request.errback)

    request.callback, request.errback = d.callback, d.errback
    return d 