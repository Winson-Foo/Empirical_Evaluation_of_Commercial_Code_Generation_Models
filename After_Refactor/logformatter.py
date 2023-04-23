import logging
import os
from typing import Any, Dict, Optional, Union
from twisted.python.failure import Failure
from scrapy import Request, Spider, signals
from scrapy.http import Response
from scrapy.utils.request import referer_str

# Constants and Configuration

SCRAPED_MSG = "Scraped from %(src)s\n%(item)s"
DROPPED_MSG = "Dropped: %(exception)s\n%(item)s"
CRAWLED_MSG = "Crawled (%(status)s) %(request)s%(request_flags)s (referer: %(referer)s)%(response_flags)s"
ITEM_ERROR_MSG = "Error processing %(item)s"
SPIDER_ERROR_MSG = "Spider error processing %(request)s (referer: %(referer)s)"
DOWNLOAD_ERROR_MSG_SHORT = "Error downloading %(request)s"
DOWNLOAD_ERROR_MSG_LONG = "Error downloading %(request)s: %(errmsg)s"

# Logger

logger = logging.getLogger(__name__)

class LogFormatter:
    """Class for generating log messages for different actions.
    """

    def crawled(self, request: Request, response: Response, spider: Spider) -> dict:
        """Logs a message when the crawler finds a webpage."""
        request_flags = f" {str(request.flags)}" if request.flags else ""
        response_flags = f" {str(response.flags)}" if response.flags else ""
        return {
            "level": logging.DEBUG,
            "msg": CRAWLED_MSG,
            "args": {
                "status": response.status,
                "request": request,
                "request_flags": request_flags,
                "referer": referer_str(request),
                "response_flags": response_flags,
                "flags": response_flags, # backward compatibility with Scrapy logformatter below 1.4 version
            },
        }

    def scraped(self, item: Any, response: Union[Response, Failure], spider: Spider) -> dict:
        """Logs a message when an item is scraped by a spider."""
        if isinstance(response, Failure):
            src = response.getErrorMessage()
        else:
            src = response
        return {
            "level": logging.DEBUG,
            "msg": SCRAPED_MSG,
            "args": {
                "src": src,
                "item": item,
            },
        }

    def dropped(self, item: Any, exception: BaseException, response: Response, spider: Spider) -> dict:
        """Logs a message when an item is dropped while it is passing through the item pipeline."""
        return {
            "level": logging.WARNING,
            "msg": DROPPED_MSG,
            "args": {
                "exception": exception,
                "item": item,
            },
        }

    def item_error(self, item: Any, exception, response: Response, spider: Spider) -> dict:
        """Logs a message when an item causes an error while it is passing through the item pipeline."""
        return {
            "level": logging.ERROR,
            "msg": ITEM_ERROR_MSG,
            "args": {
                "item": item,
            },
        }

    def spider_error(self, failure: Failure, request: Request, response: Response, spider: Spider) -> dict:
        """Logs an error message from a spider."""
        return {
            "level": logging.ERROR,
            "msg": SPIDER_ERROR_MSG,
            "args": {
                "request": request,
                "referer": referer_str(request),
            },
        }

    def download_error(self, failure: Failure, request: Request, spider: Spider, errmsg: Optional[str] = None) -> dict:
        """Logs a download error message from a spider (typically coming from the engine)."""
        args: Dict[str, Any] = {"request": request}
        if errmsg:
            msg = DOWNLOAD_ERROR_MSG_LONG
            args["errmsg"] = errmsg
        else:
            msg = DOWNLOAD_ERROR_MSG_SHORT
        return {
            "level": logging.ERROR,
            "msg": msg,
            "args": args,
        }

def _setup_logging():
    # Set up the logging configuration
    logging.basicConfig(
        level=logging.DEBUG, # change this to change logging level
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logging.getLogger("twisted").propagate = False

class CustomSpider(Spider):
    """Custom spider that uses the new log formatter for logging."""

    custom_settings = {
        'LOG_FORMATTER': 'myproject.LogFormatter',
    }
    
    def __init__(self, *args, **kwargs):
        _setup_logging() # set up the logging configuration
        super().__init__(*args, **kwargs)
        
    def start_requests(self):
        logger.info('Starting requests...')
        yield Request('https://www.example.com/', callback=self.parse)
        
    def parse(self, response):
        logger.debug('Parsing response...')
        items = response.xpath('//div[@class="item"]')
        for item in items:
            yield {
                'title': item.xpath('.//h2/text()').get().strip(),
                'description': item.xpath('.//p/text()').get().strip(),
            }
        
    def parse_item(self, item):
        try:
            # process item here...
            logger.debug('Processing item %r', item)
        except Exception as exc:
            logger.exception('Error processing item: %r', item)
            return {'__process_failure__': exc}

# Alternatively, set up the logging configuration once in the main module
# and use logging.getLogger(__name__) instead of logger in CustomSpider class.

if __name__ == '__main__':
    _setup_logging()
    
    from scrapy.crawler import CrawlerProcess
    from scrapy.utils.project import get_project_settings
    
    process = CrawlerProcess(get_project_settings())
    process.crawl(CustomSpider)
    process.start()