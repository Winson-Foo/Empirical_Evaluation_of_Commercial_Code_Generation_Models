# Extension Manager
# See documentation in docs/topics/extensions.rst
from scrapy.middleware import MiddlewareManager
from scrapy.utils.conf import build_component_list


class ExtensionManager(MiddlewareManager):

    # Set component name
    component_name = "extension"

    # Get middleware list from settings
    @classmethod
    def get_middleware(cls, settings):
        middleware_list = build_component_list(settings.getwithbase("EXTENSIONS"))
        return middleware_list