import logging
import sys

from abc import ABC, abstractmethod
from typing import Optional

from scrapy.utils.python import to_unicode


logger = logging.getLogger(__name__)


class RobotParser(ABC):
    @staticmethod
    def parse_robotstxt(
        robotstxt_body: bytes, spider: Optional[str], to_native_str_type: bool = False
    ) -> str:
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

    @classmethod
    @abstractmethod
    def from_crawler(cls, crawler: "Crawler", robotstxt_body: bytes) -> "RobotParser":
        pass

    @abstractmethod
    def is_allowed(self, url: str, user_agent: str) -> bool:
        pass


class PythonRobotParser(RobotParser):
    def __init__(self, robotstxt_body: bytes, spider: Optional[str]):
        from urllib.robotparser import RobotFileParser

        self.spider = spider
        robotstxt_body = self.parse_robotstxt(robotstxt_body, spider, to_native_str_type=True)
        self.robot_parser = RobotFileParser()
        self.robot_parser.parse(robotstxt_body.splitlines())

    @classmethod
    def from_crawler(cls, crawler: "Crawler", robotstxt_body: bytes) -> "PythonRobotParser":
        spider = None if not crawler else crawler.spider
        return cls(robotstxt_body, spider)

    def is_allowed(self, url: str, user_agent: str) -> bool:
        user_agent = to_unicode(user_agent)
        url = to_unicode(url)
        return self.robot_parser.can_fetch(user_agent, url)


class ReppyRobotParser(RobotParser):
    def __init__(self, robotstxt_body: bytes, spider: Optional[str]):
        from reppy.robots import Robots

        self.spider = spider
        self.robot_parser = Robots.parse("", robotstxt_body)

    @classmethod
    def from_crawler(cls, crawler: "Crawler", robotstxt_body: bytes) -> "ReppyRobotParser":
        spider = None if not crawler else crawler.spider
        return cls(robotstxt_body, spider)

    def is_allowed(self, url: str, user_agent: str) -> bool:
        return self.robot_parser.allowed(url, user_agent)


class RerpRobotParser(RobotParser):
    def __init__(self, robotstxt_body: bytes, spider: Optional[str]):
        from robotexclusionrulesparser import RobotExclusionRulesParser

        self.spider = spider
        robotstxt_body = self.parse_robotstxt(robotstxt_body, spider)
        self.robot_parser = RobotExclusionRulesParser()
        self.robot_parser.parse(robotstxt_body)

    @classmethod
    def from_crawler(cls, crawler: "Crawler", robotstxt_body: bytes) -> "RerpRobotParser":
        spider = None if not crawler else crawler.spider
        return cls(robotstxt_body, spider)

    def is_allowed(self, url: str, user_agent: str) -> bool:
        user_agent = to_unicode(user_agent)
        url = to_unicode(url)
        return self.robot_parser.is_allowed(user_agent, url)


class ProtegoRobotParser(RobotParser):
    def __init__(self, robotstxt_body: bytes, spider: Optional[str]):
        from protego import Protego

        self.spider = spider
        robotstxt_body = self.parse_robotstxt(robotstxt_body, spider)
        self.robot_parser = Protego.parse(robotstxt_body)

    @classmethod
    def from_crawler(cls, crawler: "Crawler", robotstxt_body: bytes) -> "ProtegoRobotParser":
        spider = None if not crawler else crawler.spider
        return cls(robotstxt_body, spider)

    def is_allowed(self, url: str, user_agent: str) -> bool:
        user_agent = to_unicode(user_agent)
        url = to_unicode(url)
        return self.robot_parser.can_fetch(url, user_agent)