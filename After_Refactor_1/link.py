from typing import List


class Link:
    """Represents an extracted link by the LinkExtractor."""

    __SLOTS__ = ["url", "text", "fragment", "nofollow"]

    def __init__(self, url: str, text: str = "", fragment: str = "", nofollow: bool = False) -> None:
        """Creates a new Link instance.

        Args:
            url: The absolute URL being linked to.
            text: The text in the anchor tag.
            fragment: The part of the URL after the hash symbol.
            nofollow: A boolean indicating the presence of a nofollow value in the rel attribute of the anchor tag.
        """
        if not isinstance(url, str):
            got = url.__class__.__name__
            raise TypeError(f"Link urls must be str objects, got {got}")
        self.url = url
        self.text = text
        self.fragment = fragment
        self.nofollow = nofollow

    def __eq__(self, other: object) -> bool:
        """Compares the equality of two Link objects."""
        if not isinstance(other, Link):
            return NotImplemented
        return (
            self.url == other.url and
            self.text == other.text and
            self.fragment == other.fragment and
            self.nofollow == other.nofollow
        )

    def __hash__(self) -> int:
        """Computes the hash value of the Link object."""
        return hash((self.url, self.text, self.fragment, self.nofollow))

    def __repr__(self) -> str:
        """Returns a string representation of the Link object."""
        return (
            f"Link(url={self.url!r}, "
            f"text={self.text!r}, "
            f"fragment={self.fragment!r}, "
            f"nofollow={self.nofollow!r})"
        )