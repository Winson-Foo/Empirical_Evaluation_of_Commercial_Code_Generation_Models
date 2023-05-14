import sys
import concurrent.futures


def read_lines(file_obj: sys.stdout, add_linebreak: bool = True, encoding: str = None) -> generator:
    """
    Return a generator that reads lines in a text file.

    :param file_obj: file object of a text file
    :param add_linebreak: whether to return a line break at the end of each line (Default: True)
    :param encoding: encoding of the text file (Default: None)
    :return: a generator that reads lines in a text file
    """

    for line in file_obj:
        if len(line) > 0:
            if encoding is None:
                yield line.strip() + ('\n' if add_linebreak else '')
            else:
                yield line.decode(encoding).strip() + ('\n' if add_linebreak else '')


def execute_in_parallel(func: callable, *args) -> list:
    """
    A wrapper function that uses the concurrent.futures module to execute a function in parallel.

    :param func: the function to execute
    :param args: the arguments to pass to the function
    :return: list of results of the function
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(func, *args))
    return results