import traceback
import warnings
from collections import defaultdict
from typing import List

from zope.interface import implementer

from scrapy.interfaces import ISpiderLoader
from scrapy.utils.misc import walk_modules
from scrapy.utils.spider import iter_spider_classes
import logging

logger = logging.getLogger(__name__)

@implementer(ISpiderLoader)
class SpiderLoader:
    def __init__(self, settings):
        self.spider_modules = settings.getlist("SPIDER_MODULES")
        self.warn_only = settings.getbool("SPIDER_LOADER_WARN_ONLY")
        self.spiders = {}
        self.found_spider_locations = defaultdict(list)
        self.load_all_spiders()

    @classmethod
    def from_settings(cls, settings):
        return cls(settings)

    def load(self, spider_name: str):
        """
        Return the Spider class for the given spider name. If the spider
        name is not found, raise a KeyError.
        """
        try:
            return self.spiders[spider_name]
        except KeyError:
            raise KeyError(f"Spider not found: {spider_name}")

    def find_by_request(self, request) -> List[str]:
        """
        Return the list of spider names that can handle the given request.
        """
        return [
            name for name, cls in self.spiders.items() if cls.handles_request(request)
        ]

    def list_spiders(self) -> List[str]:
        """
        Return a list with the names of all spiders available in the project.
        """
        return list(self.spiders.keys())

    def load_all_spiders(self):
        for module_name in self.spider_modules:
            try:
                self.load_spiders_from_module(module_name)
            except ImportError:
                if self.warn_only:
                    logger.warn(f"Could not load spiders from module '{module_name}'.")
                else:
                    raise
        self.check_for_duplicates()

    def load_spiders_from_module(self, module_name: str):
        for module in walk_modules(module_name):
            self.load_spiders_from_module_file(module)

    def load_spiders_from_module_file(self, module):
        for spider_class in iter_spider_classes(module):
            self.found_spider_locations[spider_class.name].append((module.__name__, spider_class.__name__))
            self.spiders[spider_class.name] = spider_class

    def check_for_duplicates(self):
        duplicate_spiders = []
        for spider_name, locations in self.found_spider_locations.items():
            if len(locations) > 1:
                for module_name, spider_class_name in locations:
                    duplicate_spiders.append(f"  {spider_class_name} named {spider_name!r} (in {module_name})")
        
        if duplicate_spiders:
            warn_msg = "There are several spiders with the same name:\n\n{}\n\n  This can cause unexpected behavior.".format(
                    "\n\n".join(duplicate_spiders))
            warnings.warn(
                warn_msg,
                category=UserWarning,
            ) 