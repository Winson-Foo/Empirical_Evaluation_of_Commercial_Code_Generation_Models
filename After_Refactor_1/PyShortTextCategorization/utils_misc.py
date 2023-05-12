import sys
from typing import Any, Generator, Iterable, Optional


def read_text_file(file_obj: Iterable[str], linebreak: Optional[bool] = True, encoding: Optional[str] = None) -> Generator[str, None, None]:
    """ Returns a generator that reads lines in a text file.

    :param file_obj: The text file object.
    :param linebreak: Whether to return a line break at the end of each line. Defaults to True.
    :param encoding: Encoding of the text file. Defaults to None.
    :return: A generator that reads lines in a text file.
    """
    for line in file_obj:
        if len(line) > 0:
            if encoding is None:
                yield line.strip() + ('\n' if linebreak else '')
            else:
                yield line.decode(encoding).strip() + ('\n' if linebreak else '')


class SinglePoolExecutor:
    """ A wrapper for Python `map` function. """

    def map(self, func: Any, *iterables: Iterable) -> map:
        """ Refer to Python `map` documentation.

        :param func: Function
        :param iterables: Iterables to loop
        :return: Generator for the map.
        """
        return map(func, *iterables)