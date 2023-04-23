from typing import List
from scrapy.middleware import MiddlewareManager
from scrapy.utils.conf import build_component_list
from scrapy.exceptions import NotConfigured


class CustomMiddlewareManager(MiddlewareManager):
    """
    A custom middleware manager to handle extensions in scrapy
    """

    component_name: str = "extension"

    @classmethod
    def _get_mwlist_from_settings(cls, settings) -> List:
        """
        Get a list of middleware components from scrapy settings
        """
        ext_settings = settings.getwithbase("EXTENSIONS")
        if not ext_settings:
            raise NotConfigured
        return build_component_list(ext_settings)