import logging
import sys
from abc import ABCMeta, abstractmethod
from typing import Union

from scrapy.utils.python import to_unicode


logger = logging.getLogger(__name__)


def decode_robotstxt(robotstxt_body: Union[bytes, str], spider: str, to_native_str_type: bool = False) -> str:
    """
    Decode the robots.txt file and return it as a string.

    :param robotstxt_body: The robots.txt file as bytes or str.
    :param spider: The spider name.
    :param to_native_str_type: Convert the robots.txt file to native str type if True.
    :return: The decoded robots.txt file as str.
    """
    try:
        if to_native_str_type:
            robotstxt_body = to_unicode(robotstxt_body)
        else:
            robotstxt_body = robotstxt_body.decode("utf-8")
    except UnicodeDecodeError:
        logger.warning(
            "Failure while parsing robots.txt. File either contains garbage or "
            "is in an encoding other than UTF-8, treating it as an empty file.",
            exc_info=sys.exc_info(),
            extra={"spider": spider},
        )
        robotstxt_body = ""
    return robotstxt_body


class RobotParser(metaclass=ABCMeta):
    """
    The base class for all robot parser implementations.
    """

    @classmethod
    @abstractmethod
    def from_crawler(cls, crawler, robotstxt_body):
        """
        Parse the content of a robots.txt_ file and return a new instance of the parser backend.

        :param crawler: The Scrapy crawler which made the request.
        :type crawler: :class:`~scrapy.crawler.Crawler` instance
        :param robotstxt_body: The content of a robots.txt_ file.
        :type robotstxt_body: bytes
        :return: A new instance of the robot parser backend.
        """
        pass

    @abstractmethod
    def allowed(self, url: str, user_agent: str) -> bool:
        """
        Return ``True`` if  ``user_agent`` is allowed to crawl ``url``, otherwise return ``False``.

        :param url: Absolute URL
        :type url: str
        :param user_agent: User agent
        :type user_agent: str
        :return: True if user agent allowed to crawl url, else False.
        """
        pass


class PythonRobotParser(RobotParser):
    """
    An implementation of RobotParser using Python's built-in urllib.robotparser library.
    """

    def __init__(self, robotstxt_body: str):
        from urllib.robotparser import RobotFileParser

        self.rp = RobotFileParser()
        self.rp.parse(robotstxt_body.splitlines())

    @classmethod
    def from_crawler(cls, crawler, robotstxt_body):
        return cls(decode_robotstxt(robotstxt_body, crawler.spider, to_native_str_type=True))

    def allowed(self, url: str, user_agent: str) -> bool:
        return self.rp.can_fetch(user_agent, url)


class ReppyRobotParser(RobotParser):
    """
    An implementation of RobotParser using the reppy.robots library.
    """

    def __init__(self, robotstxt_body: str):
        from reppy.robots import Robots

        self.rp = Robots.parse("", robotstxt_body)

    @classmethod
    def from_crawler(cls, crawler, robotstxt_body):
        return cls(decode_robotstxt(robotstxt_body, crawler.spider))

    def allowed(self, url: str, user_agent: str) -> bool:
        return self.rp.allowed(url, user_agent)


class RerpRobotParser(RobotParser):
    """
    An implementation of RobotParser using the robotexclusionrulesparser library.
    """

    def __init__(self, robotstxt_body: str):
        from robotexclusionrulesparser import RobotExclusionRulesParser

        self.rp = RobotExclusionRulesParser()
        self.rp.parse(robotstxt_body)

    @classmethod
    def from_crawler(cls, crawler, robotstxt_body):
        return cls(decode_robotstxt(robotstxt_body, crawler.spider))

    def allowed(self, url: str, user_agent: str) -> bool:
        return self.rp.is_allowed(user_agent, url)


class ProtegoRobotParser(RobotParser):
    """
    An implementation of RobotParser using the protego library.
    """

    def __init__(self, robotstxt_body: str):
        from protego import Protego

        self.rp = Protego.parse(robotstxt_body)

    @classmethod
    def from_crawler(cls, crawler, robotstxt_body):
        return cls(decode_robotstxt(robotstxt_body, crawler.spider))

    def allowed(self, url: str, user_agent: str) -> bool:
        return self.rp.can_fetch(url, user_agent) 