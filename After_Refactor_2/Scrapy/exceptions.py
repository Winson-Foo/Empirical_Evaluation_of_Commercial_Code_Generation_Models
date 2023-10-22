class ScrapyException(Exception):
    """Base class for all Scrapy exceptions to inherit from."""

    pass


class NotConfigured(ScrapyException):
    """Indicates a missing configuration situation"""


class _InvalidOutput(TypeError):
    """
    Indicates an invalid value has been returned by a middleware's processing method.
    Internal and undocumented, it should not be raised or caught by user code.
    """


class HttpException(ScrapyException):
    """Base class for HTTP and crawling exceptions to inherit from."""


class IgnoreRequest(HttpException):
    """Indicates a decision was made not to process a request"""


class DontCloseSpider(HttpException):
    """Request the spider not to be closed yet"""


class CloseSpider(HttpException):
    """Raise this from callbacks to request the spider to be closed"""

    def __init__(self, reason: str = "cancelled") -> None:
        super().__init__()
        self.reason = reason


class StopDownload(HttpException):
    """
    Stop the download of the body for a given response.
    The 'fail' boolean parameter indicates whether or not the resulting partial response
    should be handled by the request errback. Note that 'fail' is a keyword-only argument.
    """

    def __init__(self, *, fail: bool = True) -> None:
        super().__init__()
        self.fail = fail


class ItemException(ScrapyException):
    """Base class for item exceptions to inherit from."""


class DropItem(ItemException):
    """Drop item from the item pipeline"""


class NotSupported(ScrapyException):
    """Indicates a feature or method is not supported"""


class CommandException(ScrapyException):
    """Base class for command exceptions to inherit from."""


class UsageError(CommandException):
    """To indicate a command-line usage error"""

    def __init__(self, *args, print_help: bool = True, **kwargs) -> None:
        self.print_help = print_help
        super().__init__(*args, **kwargs)


class ScrapyDeprecationWarning(Warning):
    """Warning category for deprecated features, since the default
    DeprecationWarning is silenced on Python 2.7+
    """


class ContractFail(AssertionError):
    """Error raised in case of a failing contract""" 