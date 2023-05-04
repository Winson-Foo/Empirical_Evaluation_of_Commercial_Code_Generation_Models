signals = {
    'engine': {
        'started': object(),
        'stopped': object(),
    },
    'spider': {
        'opened': object(),
        'idle': object(),
        'closed': object(),
        'error': object(),
    },
    'request': {
        'scheduled': object(),
        'dropped': object(),
        'reached_downloader': object(),
        'left_downloader': object(),
        'received': object(),
    },
    'response': {
        'received': object(),
        'downloaded': object(),
        'headers_received': object(),
        'bytes_received': object(),
    },
    'item': {
        'scraped': object(),
        'dropped': object(),
        'error': object(),
        'passed': object(),
    },
    # for backward compatibility
    'stats': {
        'spider_opened': object(),
        'spider_closing': object(),
        'spider_closed': object(),
    }
} 