class ScrapyError(Exception):
    """Base exception class for all Scrapy exceptions."""
    pass

    
class ConfigurationError(ScrapyError):
    """Indicates a missing or incorrect configuration value."""
    pass


class InvalidMiddlewareOutput(ScrapyError):
    """Indicates an invalid value has been returned by a middleware's processing method."""
    pass


class IgnoreRequest(ScrapyError):
    """Indicates a decision was made not to process a request."""
    pass


class DontCloseSpider(ScrapyError):
    """Request the spider not to be closed yet."""
    pass


class CloseSpider(ScrapyError):
    """Raise this from callbacks to request the spider to be closed."""

    def __init__(self, reason: str = "cancelled"):
        super().__init__()
        self.reason = reason


class StopDownload(ScrapyError):
    """
    Stop the download of the body for a given response.
    The 'fail' boolean parameter indicates whether or not the resulting partial response
    should be handled by the request errback. Note that 'fail' is a keyword-only argument.
    """

    def __init__(self, *, fail=True):
        super().__init__()
        self.fail = fail


class DropItem(ScrapyError):
    """Drop item from the item pipeline."""
    pass


class FeatureNotSupported(ScrapyError):
    """Indicates a feature or method is not supported."""
    pass


class UsageError(ScrapyError):
    """
    To indicate a command-line usage error.
    The 'print_help' boolean parameter indicates whether or not to print the help message.
    Note that 'print_help' is a keyword-only argument.
    """

    def __init__(self, *a, print_help=True, **kw):
        self.print_help = print_help
        super().__init__(*a, **kw)


class ScrapyDeprecationWarning(UserWarning):
    """Warning category for deprecated features."""
    pass

    
class ContractFail(AssertionError):
    """Error raised in case of a failing contract."""
    pass
