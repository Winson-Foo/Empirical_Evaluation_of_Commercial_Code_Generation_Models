# link.py

from typing import Optional


class Link:
    """
    Link objects represent an extracted link by the LinkExtractor.
    """

    __slots__ = ["url", "text", "fragment", "nofollow"]

    def __init__(self, url: str, text: Optional[str] = None, fragment: Optional[str] = None, nofollow: bool = False):
        if not isinstance(url, str):
            got = url.__class__.__name__
            raise TypeError(f"Link urls must be str objects, got {got}")
        self.url = url
        self.text = text or ""
        self.fragment = fragment or ""
        self.nofollow = nofollow

    def __eq__(self, other):
        return (
            self.url == other.url
            and self.text == other.text
            and self.fragment == other.fragment
            and self.nofollow == other.nofollow
        )

    def __hash__(self):
        return hash(self.url) ^ hash(self.text) ^ hash(self.fragment) ^ hash(self.nofollow)

    def __repr__(self):
        return f"Link(url={self.url!r}, text={self.text!r}, fragment={self.fragment!r}, nofollow={self.nofollow!r})" 