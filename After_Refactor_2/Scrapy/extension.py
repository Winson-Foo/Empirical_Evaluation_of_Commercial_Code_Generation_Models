from scrapy.middleware import MiddlewareManager
from scrapy.utils.conf import build_component_list


class ExtensionManager(MiddlewareManager):
    COMPONENT_NAME = "extensions"

    @classmethod
    def from_settings(cls, settings):
        extensions = build_component_list(settings.getwithbase(cls.COMPONENT_NAME))
        return cls.from_crawler_settings(settings, extensions)

    @classmethod
    def from_crawler_settings(cls, settings, extensions):
        mwlist = cls._get_mwlist_from_settings(settings)
        mwlist.extend(extensions)
        middlewares = cls._get_middleware_instances(settings, mwlist)
        return cls(settings, middlewares)

    @classmethod
    def _get_mwlist_from_settings(cls, settings):
        return build_component_list(settings.getwithbase(cls.COMPONENT_NAME))

    @staticmethod
    def _get_middleware_instances(settings, mwlist):
        middlewares = []
        for mwpath in mwlist:
            try:
                mwcls = load_object(mwpath)
            except ImportError as e:
                raise ImportError(f"Error loading extension '{mwpath}': {e}")
            mwinstance = mwcls.from_crawler_settings(settings)
            middlewares.append(mwinstance)
        return middlewares

    def __init__(self, settings, middlewares):
        super().__init__(middlewares)
        self.settings = settings
        self.stats = None

    def configure_stats(self, stats):
        self.stats = stats
        for mw in self.middlewares:
            if hasattr(mw, 'configure_stats'):
                mw.configure_stats(stats) 