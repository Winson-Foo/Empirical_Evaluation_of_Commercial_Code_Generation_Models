import logging
import os
from typing import Any, Dict, Optional, Union

from scrapy import Request, Spider
from scrapy.http import Response
from scrapy.utils.request import referer_str


class LogFormatter:
    def crawled(self, request: Request, response: Response, spider: Spider) -> dict:
        request_flags = f" {str(request.flags)}" if request.flags else ""
        response_flags = f" {str(response.flags)}" if response.flags else ""
        return {
            "level": logging.DEBUG,
            "msg": f"Crawled ({response.status}) {request}{request_flags} (referer: {referer_str(request)}){response_flags}",
        }

    def scraped(
        self, item: Any, response: Union[Response, Exception], spider: Spider
    ) -> dict:
        if isinstance(response, Exception):
            src = response.getErrorMessage()
        else:
            src = response
        return {
            "level": logging.DEBUG,
            "msg": f"Scraped from {src}{os.linesep}{item}",
        }

    def dropped(
        self, item: Any, exception: Exception, response: Response, spider: Spider
    ) -> dict:
        return {
            "level": logging.WARNING,
            "msg": f"Dropped: {exception}{os.linesep}{item}",
        }

    def item_error(
        self, item: Any, exception: Exception, response: Response, spider: Spider
    ) -> dict:
        return {
            "level": logging.ERROR,
            "msg": f"Error processing {item}",
        }

    def spider_error(
        self, failure: Exception, request: Request, response: Response, spider: Spider
    ) -> dict:
        return {
            "level": logging.ERROR,
            "msg": f"Spider error processing {request} (referer: {referer_str(request)})",
        }

    def download_error(
        self,
        failure: Exception,
        request: Request,
        spider: Spider,
        errmsg: Optional[str] = None,
    ) -> dict:
        if errmsg:
            msg = f"Error downloading {request}: {errmsg}"
        else:
            msg = f"Error downloading {request}"
        return {
            "level": logging.ERROR,
            "msg": msg,
        }

    @classmethod
    def from_crawler(cls, crawler):
        return cls() 