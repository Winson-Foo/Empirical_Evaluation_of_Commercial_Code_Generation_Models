import logging
import os
from typing import Any, Dict, Optional, Union
from twisted.python.failure import Failure

from scrapy import Request, Spider
from scrapy.http import Response
from scrapy.utils.request import referer_str


class LogFormatter:
    """Class for generating log messages for different actions."""

    @classmethod
    def from_crawler(cls, crawler) -> "LogFormatter":
        return cls()

    def crawled(self, request: Request, response: Response, spider: Spider) -> dict:
        """Logs a message when the crawler finds a webpage."""
        request_flags = f" {str(request.flags)}" if request.flags else ""
        response_flags = f" {str(response.flags)}" if response.flags else ""
        args = {
            "status": response.status,
            "request": request,
            "request_flags": request_flags,
            "referer": referer_str(request),
            "response_flags": response_flags,
            # backward compatibility with Scrapy logformatter below 1.4 version
            "flags": response_flags,
        }
        msg = f"Crawled ({response.status}) {request}{request_flags}" \
            f" (referer: {referer_str(request)}){response_flags}"
        return {"level": logging.DEBUG, "msg": msg, "args": args}

    def scraped(
        self, item: Any, response: Union[Response, Failure], spider: Spider
    ) -> dict:
        """Logs a message when an item is scraped by a spider."""
        src = response.getErrorMessage() if isinstance(response, Failure) else response
        args = {"src": src, "item": item}
        msg = f"Scraped from {src}{os.linesep}{item}"
        return {"level": logging.DEBUG, "msg": msg, "args": args}

    def dropped(
        self, item: Any, exception: BaseException, response: Response, spider: Spider
    ) -> dict:
        """Logs a message when an item is dropped while passing through the item pipeline."""
        args = {"exception": exception, "item": item}
        msg = f"Dropped: {exception}{os.linesep}{item}"
        return {"level": logging.WARNING, "msg": msg, "args": args}

    def item_error(
        self, item: Any, exception, response: Response, spider: Spider
    ) -> dict:
        """Logs a message when an item causes an error while passing through the item pipeline."""
        args = {"item": item}
        msg = f"Error processing {item}"
        return {"level": logging.ERROR, "msg": msg, "args": args}

    def spider_error(
        self, failure: Failure, request: Request, response: Response, spider: Spider
    ) -> dict:
        """Logs an error message from a spider."""
        args = {"request": request, "referer": referer_str(request)}
        msg = f"Spider error processing {request} (referer: {referer_str(request)})"
        return {"level": logging.ERROR, "msg": msg, "args": args}

    def download_error(
        self,
        failure: Failure,
        request: Request,
        spider: Spider,
        errmsg: Optional[str] = None,
    ) -> dict:
        """Logs a download error message from a spider (typically coming from the engine)."""
        args = {"request": request}
        if errmsg:
            msg = f"Error downloading {request}: {errmsg}"
            args["errmsg"] = errmsg
        else:
            msg = f"Error downloading {request}"
        return {"level": logging.ERROR, "msg": msg, "args": args} 