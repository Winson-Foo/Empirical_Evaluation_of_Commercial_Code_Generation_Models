SIGNALS = {
    'engine_started': engine_started,
    'engine_stopped': engine_stopped,
    'spider_opened': spider_opened,
    'spider_idle': spider_idle,
    'spider_closed': spider_closed,
    'spider_error': spider_error,
    'request_scheduled': request_scheduled,
    'request_dropped': request_dropped,
    'request_reached_downloader': request_reached_downloader,
    'request_left_downloader': request_left_downloader,
    'response_received': response_received,
    'response_downloaded': response_downloaded,
    'headers_received': headers_received,
    'bytes_received': bytes_received,
    'item_scraped': item_scraped,
    'item_dropped': item_dropped,
    'item_error': item_error,
    'stats_spider_opened': spider_opened,
    'stats_spider_closing': spider_closed,
    'stats_spider_closed': spider_closed,
    'item_passed': item_scraped,
    'request_received': request_scheduled
}

# documented signals
# see docs/topics/signals.rst