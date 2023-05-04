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
    Spider loader is a class which locates and loads spiders in a Scrapy project.
    """

    def __init__(self, settings):
        self.spider_modules = settings.getlist("SPIDER_MODULES")
        self.warn_only = settings.getbool("SPIDER_LOADER_WARN_ONLY")
        self._spiders = {}
        self._found_spiders = defaultdict(list)
        self._load_all_spiders()

    @classmethod
    def from_settings(cls, settings):
        """
        Create a spider loader instance from Scrapy settings object.
        """
        return cls(settings)

    def load(self, spider_name):
        """
        Return the Spider class for the given spider name. If the spider
        name is not found, raise a KeyError.
        """
        try:
            return self._spiders[spider_name]
        except KeyError:
            raise KeyError(f"Spider not found: {spider_name}")

    def find_by_request(self, request):
        """
        Return the list of spider names that can handle the given request.
        """
        return [
            name for name, cls in self._spiders.items() if cls.handles_request(request)
        ]

    def list_spiders(self):
        """
        Return a list with the names of all spiders available in the project.
        """
        return list(self._spiders.keys())

    def _check_name_duplicates(self):
        """
        Check for duplicate spider names, raise warning if found.
        """
        duplicate_spiders = []
        for name, locations in self._found_spiders.items():
            if len(locations) > 1:
                for mod, cls in locations:
                    duplicate_spiders.append(
                        f"  {cls} named {name!r} (in {mod})"
                    )

        if duplicate_spiders:
            duplicate_spiders_string = "\n\n".join(duplicate_spiders)
            warnings.warn(
                "There are several spiders with the same name:\n\n"
                f"{duplicate_spiders_string}\n\n  This can cause unexpected behavior.",
                category=UserWarning,
            )

    def _load_spiders(self, module):
        """
        Load spiders from a module and add them to spider registry.
        """
        for spider_class in iter_spider_classes(module):
            self._found_spiders[spider_class.name].append(
                (module.__name__, spider_class.__name__)
            )
            self._spiders[spider_class.name] = spider_class

    def _load_all_spiders(self):
        """
        Load all spiders from the spider modules and check for duplicate names.
        """
        for module_name in self.spider_modules:
            try:
                for module in walk_modules(module_name):
                    self._load_spiders(module)
            except ImportError:
                if self.warn_only:
                    warnings.warn(
                        f"\n{traceback.format_exc()}Could not load spiders "
                        f"from module '{module_name}'. "
                        "See above traceback for details.",
                        category=RuntimeWarning,
                    )
                else:
                    raise
        self._check_name_duplicates() 