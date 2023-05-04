import logging
import os
from typing import Any, Dict, Optional, Union
from twisted.python.failure import Failure
from scrapy import Request, Spider
from scrapy.http import Response
from scrapy.utils.request import referer_str

SCRAPEDMSG = "Scraped from {src}" + os.linesep + "{item}"
DROPPEDMSG = "Dropped: {exception}" + os.linesep + "{item}"
CRAWLEDMSG = "Crawled ({status}) {request}{request_flags} (referer: {referer}){response_flags}"
ITEMERRORMSG = "Error processing {item}"
SPIDERERRORMSG = "Spider error processing {request} (referer: {referer})"
DOWNLOADERRORMSG_SHORT = "Error downloading {request}"
DOWNLOADERRORMSG_LONG = "Error downloading {request}: {errmsg}"

LOG_LEVEL_DEBUG = logging.DEBUG
LOG_LEVEL_ERROR = logging.ERROR
LOG_LEVEL_WARNING = logging.WARNING

class LogFormatter:
    """Class for generating log messages for different actions."""

    def crawled(self, request: Request, response: Response, spider: Spider) -> dict:
        request_flags = f" {str(request.flags)}" if request.flags else ""
        response_flags = f" {str(response.flags)}" if response.flags else ""
        return {
            "level": LOG_LEVEL_DEBUG,
            "msg": CRAWLEDMSG,
            "args": {
                "status": response.status,
                "request": request,
                "request_flags": request_flags,
                "referer": referer_str(request),
                "response_flags": response_flags,
                "flags": response_flags,
            },
        }

    def scraped(
        self, item: Any, response: Union[Response, Failure], spider: Spider
    ) -> dict:
        src: Any = response.getErrorMessage() if isinstance(response, Failure) else response
        return {
            "level": LOG_LEVEL_DEBUG,
            "msg": SCRAPEDMSG,
            "args": {
                "src": src,
                "item": item,
            },
        }

    def dropped(
        self, item: Any, exception: BaseException, response: Response, spider: Spider
    ) -> dict:
        return {
            "level": LOG_LEVEL_WARNING,
            "msg": DROPPEDMSG,
            "args": {
                "exception": exception,
                "item": item,
            },
        }

    def item_error(
        self, item: Any, exception, response: Response, spider: Spider
    ) -> dict:
        return {
            "level": LOG_LEVEL_ERROR,
            "msg": ITEMERRORMSG,
            "args": {
                "item": item,
            },
        }

    def spider_error(
        self, failure: Failure, request: Request, response: Response, spider: Spider
    ) -> dict:
        return {
            "level": LOG_LEVEL_ERROR,
            "msg": SPIDERERRORMSG,
            "args": {
                "request": request,
                "referer": referer_str(request),
            },
        }

    def download_error(
        self,
        failure: Failure,
        request: Request,
        spider: Spider,
        errmsg: Optional[str] = None,
    ) -> dict:
        args: Dict[str, Any] = {"request": request}
        if errmsg:
            msg = DOWNLOADERRORMSG_LONG
            args["errmsg"] = errmsg
        else:
            msg = DOWNLOADERRORMSG_SHORT
        return {
            "level": LOG_LEVEL_ERROR,
            "msg": msg,
            "args": args,
        }

    @classmethod
    def from_crawler(cls, crawler) -> "LogFormatter":
        return cls() 