from zope.interface import Interface


class ISpiderLoader(Interface):
    """
    An interface defining methods for loading and managing spiders.
    """
    @staticmethod
    def from_settings(settings):
        """
        Return an instance of the class for the given settings
        """
        pass

    @staticmethod
    def load(spider_name):
        """
        Return the Spider class for the given spider name.
        If the spider name is not found, it must raise a KeyError.
        """
        pass

    @staticmethod
    def list():
        """
        Return a list with the names of all spiders available in the project
        """
        pass

    @staticmethod
    def find_by_request(request):
        """
        Return the list of spiders names that can handle the given request
        """
        pass