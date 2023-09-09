import sys


def read_lines(file, linebreak=True, encoding=None):
    """
    Returns a generator that reads lines from a text file.
    :param file: A file object of a text file
    :param linebreak: Whether to return a line break at the end of each line (default: True)
    :param encoding: The encoding of the text file (default: None)
    :return: A generator that reads lines from a text file
    :rtype: generator
    """
    for line in file:
        if encoding is not None:
            line = line.decode(encoding)
        yield line.strip() + ("\n" if linebreak else "")


class SinglePoolExecutor:
    """
    A wrapper for Python `map` function.
    """
    def map(self, func, *iterables):
        """
        Refer to Python `map` documentation.
        :param func: A function
        :param iterables: Iterables to loop
        :return: A generator for the map
        :rtype: map
        """
        return map(func, *iterables)