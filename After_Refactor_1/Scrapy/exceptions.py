# Internal exceptions

class NotConfigured(Exception):
    pass

class _InvalidOutput(TypeError):
    pass

# HTTP and crawling exceptions

class IgnoreRequest(Exception):
    pass

class DontCloseSpider(Exception):
    pass

class CloseSpider(Exception):
    def __init__(self, reason: str = "cancelled"):
        super().__init__()
        self.reason = reason

class StopDownload(Exception):
    def __init__(self, *, fail=True):
        super().__init__()
        self.fail = fail

# Item exceptions

class DropItem(Exception):
    pass

class NotSupported(Exception):
    pass

# Command exceptions

class UsageError(Exception):
    def __init__(self, *a, **kw):
        self.print_help = kw.pop("print_help", True)
        super().__init__(*a, **kw)

class ScrapyDeprecationWarning(Warning):
    pass

class ContractFail(AssertionError):
    pass

# Documented exceptions are in docs/topics/exceptions.rst. 
# Please don't add new exceptions without documenting them there.