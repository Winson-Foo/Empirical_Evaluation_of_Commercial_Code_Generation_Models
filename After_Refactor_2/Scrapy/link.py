from typing import Optional, NamedTuple

Link = NamedTuple('Link', [('url', str), ('text', str), ('fragment', str), ('nofollow', bool)])

def make_link(url: str, text: str='', fragment: Optional[str]=None, nofollow: bool=False) -> Link:
    """Create a Link named tuple with the given attributes.

    Args:
        url (str): The absolute url being linked to in the anchor tag.
        text (str, optional): The text in the anchor tag. Defaults to ''.
        fragment (str, optional): The part of the url after the hash symbol. Defaults to None.
        nofollow (bool, optional): Indication of the presence or absense of 'nofollow' in the rel attribute.
    Returns:
        Link: A Link named tuple.
    """
    if not isinstance(url, str):
        raise TypeError(f"Link urls must be str objects, got {type(url).__name__}")
    return Link(url, text, fragment or '', nofollow) 