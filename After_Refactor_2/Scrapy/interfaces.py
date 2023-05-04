from zope.interface import Interface
from typing import List

class ISpiderLoader(Interface):
    """Interface for loading, listing and finding spiders"""

    @classmethod
    def from_settings(cls, settings: dict) -> 'ISpiderLoader':
        """Return an instance of the class for the given settings"""

    @classmethod
    def load(cls, spider_name: str) -> object:
        """Return the Spider class for the given spider name.
        
        Args:
            spider_name (str): The name of the spider.

        Raises:
            KeyError: If the spider name is not found.

        Returns:
            object: The Spider class for the given spider name.
        """

    @classmethod
    def list(cls) -> List[str]:
        """Return a list with the names of all spiders available in the project.

        Returns:
            List[str]: List of all spider names.
        """

    @classmethod
    def find_by_request(cls, request: object) -> List[str]:
        """Return the list of spiders names that can handle the given request.
        
        Args:
            request (object): The request object to be handled by spiders.

        Returns:
            List[str]: List of spider names that can handle the given request.
        """ 