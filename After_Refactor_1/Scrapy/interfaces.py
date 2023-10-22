from typing import List


class ISpiderLoader:
    @staticmethod
    def from_settings(settings: dict) -> 'ISpiderLoader':
        """Return an instance of the class for the given settings"""

    @classmethod
    def load(cls, spider_name: str) -> 'Spider':
        """Return the Spider class for the given spider name.
        If the spider name is not found, it must raise a KeyError."""

    @classmethod
    def list(cls) -> List[str]:
        """Return a list with the names of all spiders available in the project"""

    @classmethod
    def find_by_request(cls, request: 'Request') -> List[str]:
        """Return the list of spiders names that can handle the given request"""


class SpiderLoaderFromSettings:
    @staticmethod
    def from_settings(settings: dict) -> 'SpiderLoaderFromSettings':
        # implementation


class SpiderLoaderLoader:
    @classmethod
    def load(cls, spider_name: str) -> 'Spider':
        # implementation


class SpiderLoaderList:
    @classmethod
    def list(cls) -> List[str]:
        # implementation


class SpiderLoaderFindByRequest:
    @classmethod
    def find_by_request(cls, request: 'Request') -> List[str]:
        # implementation


class SpiderLoader(ISpiderLoader, SpiderLoaderFromSettings, SpiderLoaderLoader, SpiderLoaderList, SpiderLoaderFindByRequest):
    pass