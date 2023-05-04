import logging
import sys
from abc import ABCMeta, abstractmethod
from typing import Optional, Tuple

from scrapy.utils.python import to_unicode
from urllib.robotparser import RobotFileParser

logger = logging.getLogger(__name__)


class RobotParser(metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def from_crawler(cls, crawler, robotstxt_body: bytes) -> 'RobotParser':
        """Parse the content of a robots.txt_ file as bytes. This must be a class method.
        It must return a new instance of the parser backend.

        :param crawler: crawler which made the request
        :type crawler: :class:`~scrapy.crawler.Crawler` instance

        :param robotstxt_body: content of a robots.txt_ file.
        :type robotstxt_body: bytes
        """
        pass

    @abstractmethod
    def allowed(self, url: str, user_agent: str) -> bool:
        """Return ``True`` if  ``user_agent`` is allowed to crawl ``url``, otherwise return ``False``.

        :param url: Absolute URL
        :type url: str

        :param user_agent: User agent
        :type user_agent: str
        """
        pass


def decode_robotstxt(robotstxt_body: bytes, spider: Optional[str] = None, to_native_str_type: bool = False) -> str:
    """Decode the robots.txt body and return as string

    :param robotstxt_body: robots.txt body as bytes
    :type robotstxt_body: bytes

    :param spider: Name of the spider
    :type spider: Optional[str]

    :param to_native_str_type: Whether to convert to native string type, defaults to False
    :type to_native_str_type: bool, optional

    :return: decoded robots.txt body as string
    :rtype: str
    """
    try:
        if to_native_str_type:
            robotstxt_body = to_unicode(robotstxt_body)
        else:
            robotstxt_body = robotstxt_body.decode("utf-8")
    except UnicodeDecodeError:
        # If we found garbage or robots.txt in an encoding other than UTF-8, disregard it.
        # Switch to 'allow all' state.
        logger.warning(
            "Failure while parsing robots.txt. File either contains garbage or "
            "is in an encoding other than UTF-8, treating it as an empty file.",
            exc_info=sys.exc_info(),
            extra={"spider": spider},
        )
        robotstxt_body = ""
    return robotstxt_body


class PythonRobotParser(RobotParser):
    def __init__(self, robotstxt_body: bytes, spider: Optional[str] = None):
        self.spider = spider
        robotstxt_body = decode_robotstxt(
            robotstxt_body, spider, to_native_str_type=True
        )
        self.rp = RobotFileParser()
        self.rp.parse(robotstxt_body.splitlines())

    @classmethod
    def from_crawler(cls, crawler, robotstxt_body: bytes) -> 'PythonRobotParser':
        spider = None if not crawler else crawler.spider
        return cls(robotstxt_body, spider)

    def allowed(self, url: str, user_agent: str) -> bool:
        user_agent = to_unicode(user_agent)
        url = to_unicode(url)
        return self.rp.can_fetch(user_agent, url)


class ReppyRobotParser(RobotParser):
    def __init__(self, robotstxt_body: bytes, spider: Optional[str] = None):
        self.spider = spider
        self.rp = Robots.parse('', robotstxt_body)

    @classmethod
    def from_crawler(cls, crawler, robotstxt_body: bytes) -> 'ReppyRobotParser':
        spider = None if not crawler else crawler.spider
        return cls(robotstxt_body, spider)

    def allowed(self, url: str, user_agent: str) -> bool:
        return self.rp.allowed(url, user_agent)


class RerpRobotParser(RobotParser):
    def __init__(self, robotstxt_body: bytes, spider: Optional[str] = None):
        self.spider = spider
        self.rp = RobotExclusionRulesParser()
        robotstxt_body = decode_robotstxt(robotstxt_body, spider)
        self.rp.parse(robotstxt_body)

    @classmethod
    def from_crawler(cls, crawler, robotstxt_body: bytes) -> 'RerpRobotParser':
        spider = None if not crawler else crawler.spider
        return cls(robotstxt_body, spider)

    def allowed(self, url: str, user_agent: str) -> bool:
        user_agent = to_unicode(user_agent)
        url = to_unicode(url)
        return self.rp.is_allowed(user_agent, url)


class ProtegoRobotParser(RobotParser):
    def __init__(self, robotstxt_body: bytes, spider: Optional[str] = None):
        self.spider = spider
        robotstxt_body = decode_robotstxt(robotstxt_body, spider)
        self.rp = Protego.parse(robotstxt_body)

    @classmethod
    def from_crawler(cls, crawler, robotstxt_body: bytes) -> 'ProtegoRobotParser':
        spider = None if not crawler else crawler.spider
        return cls(robotstxt_body, spider)

    def allowed(self, url: str, user_agent: str) -> bool:
        user_agent = to_unicode(user_agent)
        url = to_unicode(url)
        return self.rp.can_fetch(url, user_agent) 