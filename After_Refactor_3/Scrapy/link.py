from typing import List


class Link:
    """Link objects represent an extracted link by the LinkExtractor.

    Attributes:
        url (str): the absolute url being linked to in the anchor tag.
        text (str): the text in the anchor tag.
        fragment (str): the part of the url after the hash symbol.
        nofollow (bool): an indication of the presence or absence of a nofollow value in the rel attribute of the anchor tag.
    """

    __slots__ = ["url", "text", "fragment", "nofollow"]

    def __init__(self, url: str, text: str = "", fragment: str = "", nofollow: bool = False) -> None:
        """Initialize a new Link object."""
        if not isinstance(url, str):
            got = url.__class__.__name__
            raise TypeError(f"Link urls must be str objects, got {got}")
        self.url = url
        self.text = text
        self.fragment = fragment
        self.nofollow = nofollow

    def __eq__(self, other: object) -> bool:
        """Check if this Link object is equal to another."""
        if isinstance(other, Link):
            return (
                self.url == other.url
                and self.text == other.text
                and self.fragment == other.fragment
                and self.nofollow == other.nofollow
            )
        return False

    def __hash__(self) -> int:
        """Generate a hash value for this Link object."""
        return hash(self.url) ^ hash(self.text) ^ hash(self.fragment) ^ hash(self.nofollow)

    def __repr__(self) -> str:
        """Return a string representation of this Link object."""
        return (
            f"Link(url={self.url!r}, text={self.text!r}, "
            f"fragment={self.fragment!r}, nofollow={self.nofollow!r})"
        )

    def __str__(self) -> str:
        """Return a string representation of this Link object."""
        return f"{self.text} ({self.url})"


def extractLinks(html: str) -> List[Link]:
    """Extract all links from an HTML string and return them as a list of Link objects."""
    # TODO: Implement link extraction
    pass 