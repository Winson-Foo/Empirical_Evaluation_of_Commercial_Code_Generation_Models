import logging
from abc import ABCMeta, abstractmethod
from typing import Optional

from scrapy.utils.python import to_unicode

logger = logging.getLogger(__name__)


def decode_robotstxt(
    robotstxt_body: bytes,
    spider_name: Optional[str] = None,
    to_native_str_type: bool = False,
) -> str:
    try:
        if to_native_str_type:
            robotstxt_body = to_unicode(robotstxt_body)
        else:
            robotstxt_body = robotstxt_body.decode("utf-8")
    except UnicodeDecodeError:
        logger.error(
            "Failed to parse robots.txt. File either contains garbage or is in an encoding "
            "other than UTF-8. Treating it as an empty file.",
            extra={"spider": spider_name},
        )
        robotstxt_body = ""
    return robotstxt_body


class RobotParser(metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def from_crawler(cls, crawler, robotstxt_body):
        pass

    @abstractmethod
    def allowed(self, url: str, user_agent: str) -> bool:
        pass


class BaseRobotParser(RobotParser):
    def __init__(self, robotstxt_body: str):
        self.robotstxt_body = robotstxt_body

    @classmethod
    def from_crawler(cls, crawler, robotstxt_body):
        spider_name = crawler.spider.name if crawler else None
        return cls(decode_robotstxt(robotstxt_body, spider_name, True))


class PythonRobotParser(BaseRobotParser):
    def __init__(self, robotstxt_body: str):
        super().__init__(robotstxt_body)
        from urllib.robotparser import RobotFileParser

        self.rp = RobotFileParser()
        self.rp.parse(self.robotstxt_body.splitlines())

    def allowed(self, url: str, user_agent: str) -> bool:
        return self.rp.can_fetch(user_agent, url)


class ReppyRobotParser(BaseRobotParser):
    def __init__(self, robotstxt_body: str):
        super().__init__(robotstxt_body)
        from reppy.robots import Robots

        self.rp = Robots.parse("", self.robotstxt_body)

    def allowed(self, url: str, user_agent: str) -> bool:
        return self.rp.allowed(url, user_agent)


class RerpRobotParser(BaseRobotParser):
    def __init__(self, robotstxt_body: str):
        super().__init__(robotstxt_body)
        from robotexclusionrulesparser import RobotExclusionRulesParser

        self.rp = RobotExclusionRulesParser()
        self.rp.parse(self.robotstxt_body)

    def allowed(self, url: str, user_agent: str) -> bool:
        return self.rp.is_allowed(user_agent, url)


class ProtegoRobotParser(BaseRobotParser):
    def __init__(self, robotstxt_body: str):
        super().__init__(robotstxt_body)
        from protego import Protego

        self.rp = Protego.parse(self.robotstxt_body)

    def allowed(self, url: str, user_agent: str) -> bool:
        return self.rp.can_fetch(url, user_agent) 