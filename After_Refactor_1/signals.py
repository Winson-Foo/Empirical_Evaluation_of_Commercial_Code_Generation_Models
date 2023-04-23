from enum import Enum

class ScrapySignal(Enum):
    ENGINE_STARTED = "engine_started"
    ENGINE_STOPPED = "engine_stopped"
    SPIDER_OPENED = "spider_opened"
    SPIDER_IDLE = "spider_idle"
    SPIDER_CLOSED = "spider_closed"
    SPIDER_ERROR = "spider_error"
    REQUEST_SCHEDULED = "request_scheduled"
    REQUEST_DROPPED = "request_dropped"
    REQUEST_REACHED_DOWNLOADER = "request_reached_downloader"
    REQUEST_LEFT_DOWNLOADER = "request_left_downloader"
    RESPONSE_RECEIVED = "response_received"
    RESPONSE_DOWNLOADED = "response_downloaded"
    HEADERS_RECEIVED = "headers_received"
    BYTES_RECEIVED = "bytes_received"
    ITEM_SCRAPED = "item_scraped"
    ITEM_DROPPED = "item_dropped"
    ITEM_ERROR = "item_error"
    ITEM_PASSED = "item_scraped" # for backward compatibility
    REQUEST_RECEIVED = "request_scheduled"

    STATS_SPIDER_OPENED = "spider_opened" # for backward compatibility
    STATS_SPIDER_CLOSING = "spider_closed" # for backward compatibility
    STATS_SPIDER_CLOSED = "spider_closed" # for backward compatibility

from scrapy import signals
from my_project import ScrapySignal

def my_callback(sender, **kwargs):
    pass
    
# connect my_callback to the SPIDER_CLOSED signal
signals.connect(my_callback, signal=ScrapySignal.SPIDER_CLOSED)