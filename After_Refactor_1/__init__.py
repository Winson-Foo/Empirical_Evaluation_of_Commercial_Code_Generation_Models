import sys
import warnings

from scrapy.http import FormRequest, Request
from scrapy.item import Field, Item
from scrapy.selector import Selector
from scrapy.spiders import Spider

# Scrapy version
VERSION = "1.0.0"

__all__ = [
    "__version__",
    "twisted_version",
    "Spider",
    "Request",
    "FormRequest",
    "Selector",
    "Item",
    "Field",
]

__version__ = VERSION
twisted_version = (19, 10, 0)


# Check minimum required Python version
if sys.version_info < (3, 7):
    raise RuntimeError(f"Scrapy {__version__} requires Python 3.7+")

# Ignore noisy twisted deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="twisted")