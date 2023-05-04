import traceback
import warnings
from collections import defaultdict
from zope.interface import implementer
from scrapy.interfaces import ISpiderLoader
from scrapy.utils.misc import walk_modules
from scrapy.utils.spider import iter_spider_classes


@implementer(ISpiderLoader)
class ScrapySpiderLoader:
    """
    ScrapySpiderLoader is a class which locates and loads spiders
    in a Scrapy project.
    """

    def __init__(self, settings):
        self.spider_modules = settings.getlist("SPIDER_MODULES")
        self.warn_only = settings.getbool("SPIDER_LOADER_WARN_ONLY")
        self.spiders = self._load_all_spiders()

    def _check_name_duplicates(self):
        duplicates = []
        for name, locations in self.found.items():
            if len(locations) > 1:
                for mod, cls in locations:
                    duplicates.append(f"{cls} named {name!r} (in {mod})")

        if duplicates:
            duplicates_string = "\n\n".join(duplicates)
            warnings.warn(f"There are several spiders with the same name:\n\n"
                          f"{duplicates_string}\n\nThis can cause unexpected behavior.",
                          category=UserWarning)

    def _load_spiders(self, module):
        spiders = []
        for spider_cls in iter_spider_classes(module):
            spiders.append((
                spider_cls.name,
                (module.__name__, spider_cls.__name__)
            ))
        return spiders

    def _load_all_spiders(self):
        self.found = defaultdict(list)
        spiders = []
        for module_name in self.spider_modules:
            try:
                for module in walk_modules(module_name):
                    spiders.extend(self._load_spiders(module))
            except ImportError:
                if self.warn_only:
                    warnings.warn(f"\n{traceback.format_exc()}Could not load spiders "
                                  f"from module '{module_name}'. "
                                  "See above traceback for details.",
                                  category=RuntimeWarning)
                else:
                    raise
        for spider_name, spider_location in spiders:
            self.found[spider_name].append(spider_location)
        self._check_name_duplicates()
        return {name: cls for name, cls in spiders}

    @classmethod
    def from_settings(cls, settings):
        return cls(settings)

    def load(self, spider_name):
        """
        Return the Spider class for the given spider name. If the spider
        name is not found, raise a KeyError.
        """
        if spider_name not in self.spiders:
            raise KeyError(f"Spider not found: {spider_name}")
        return self.spiders[spider_name]

    def find_by_request(self, request):
        """
        Return the list of spider names that can handle the given request.
        """
        return [name for name, cls in self.spiders.items() if cls.handles_request(request)]

    def list(self):
        """
        Return a list with the names of all spiders available in the project.
        """
        return list(self.spiders.keys()) 