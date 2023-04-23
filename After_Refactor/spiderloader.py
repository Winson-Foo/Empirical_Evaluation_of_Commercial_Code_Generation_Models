import traceback
import warnings
from collections import defaultdict

from zope.interface import implementer

from scrapy.interfaces import ISpiderLoader
from scrapy.utils.misc import walk_modules
from scrapy.utils.spider import iter_spider_classes


@implementer(ISpiderLoader)
class SpiderLoader:
    """
    SpiderLoader is a class which locates and loads spiders
    in a Scrapy project.
    """

    def __init__(self, settings):
        self.spider_modules = settings.getlist("SPIDER_MODULES")
        self.warn_only = settings.getbool("SPIDER_LOADER_WARN_ONLY")
        self.spider_classes = {}
        self._found_spiders = defaultdict(list)
        self._load_all_spiders()
        self._check_name_duplicates()

    def _load_spiders(self, module):
        for spider_class in iter_spider_classes(module):
            self._found_spiders[spider_class.name].append((module.__name__, spider_class.__name__))
            self.spider_classes[spider_class.name] = spider_class

    def _load_all_spiders(self):
        for module_name in self.spider_modules:
            try:
                for module in walk_modules(module_name):
                    self._load_spiders(module)
            except ImportError as exc:
                if self.warn_only:
                    warnings.warn(
                        f"Could not load spiders from module '{module_name}'. "
                        f"Error: {exc!r}",
                        category=RuntimeWarning,
                    )
                else:
                    raise

    def _check_name_duplicates(self):
        duplicate_spiders = [f"  {cls} named {name!r} (in {mod})"
                             for name, locations in self._found_spiders.items()
                             for mod, cls in locations
                             if len(locations) > 1]
        if duplicate_spiders:
            message = "There are several spiders with the same name:\n\n" + "\n\n".join(duplicate_spiders) \
                      + "\n\n  This can cause unexpected behavior."
            warnings.warn(message, category=UserWarning)

    @classmethod
    def from_settings(cls, settings):
        return cls(settings)

    def load(self, spider_name):
        """
        Return the Spider class for the given spider name. If the spider
        name is not found, raise a KeyError.
        """
        try:
            return self.spider_classes[spider_name]
        except KeyError:
            raise KeyError(f"Spider not found: {spider_name}")

    def find_by_request(self, request):
        """
        Return the list of spider names that can handle the given request.
        """
        return [
            name for name, cls in self.spider_classes.items() if cls.handles_request(request)
        ]

    def list(self):
        """
        Return a list with the names of all spiders available in the project.
        """
        return list(self.spider_classes.keys())