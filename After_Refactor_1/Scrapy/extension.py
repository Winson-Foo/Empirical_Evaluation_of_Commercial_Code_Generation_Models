from scrapy.middleware import MiddlewareManager
from scrapy.utils.conf import build_component_list
from typing import List


class ExtensionManager(MiddlewareManager):
    """
    Manages the extensions for a Scrapy spider.
    """
    component_name: str = "extension"

    @classmethod
    def get_extension_list_from_settings(cls, settings: dict) -> List[str]:
        """
        Returns a list of extensions from the Scrapy settings.
        """
        EXTENSIONS_KEY: str = "EXTENSIONS"
        return build_component_list(settings.getwithbase(EXTENSIONS_KEY)) 