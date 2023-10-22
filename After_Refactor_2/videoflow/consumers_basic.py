import json
from typing import Any, Optional
import requests
from ..core.node import ConsumerNode


class CommandlineConsumer(ConsumerNode):
    '''Writes the input received to the command line.'''

    def __init__(self, sep: str = ' ', end: str = '\n') -> None:
        """
        Initialize CommandlineConsumer class.

        Args:
            sep: Separator to use between tokens. Default is a single space.
            end: End of line character. Default is a new line.
        """
        super().__init__()
        self._end = end
        self._sep = sep
    
    def consume(self, item: Any) -> None:
        '''
        Prints `item` to the command line, adding an end of line character after it.

        Args:
            item: It can be anything that can be printed with the ``print()`` function
        '''
        print(item, sep=self._sep, end=self._end)


class VoidConsumer(ConsumerNode):
    '''Ignores the input received. Helpful in debugging flows.'''

    def __init__(self) -> None:
        super().__init__()
    
    def consume(self, item: Any) -> None:
        '''Does nothing with the item passed.'''
        pass


class WebhookConsumer(ConsumerNode):
    '''
    Posts data to a webhook URL.

    Args:
        host: The URL of the webhook.
        method: The HTTP method to use. Default is POST.
    '''

    def __init__(self, host: str, method: str = "POST") -> None:
        super().__init__()
        self.host = host
        self.method = method

    def consume(self, item: Any) -> None:
        '''
        Makes a POST request with the given data.

        Args:
            item: Data to post to the webhook. Should be json serializable.
        '''
        try:
            item = json.dumps(item)
        except TypeError:
            print("Unable to convert to JSON format.")
            return

        requests.post(self.host, data=item, method=self.method)


class FileAppenderConsumer(ConsumerNode):
    '''
    Appends data to a file.

    Args:
        file: Path to the file to append to.
        mode: Mode to open the file in. Default is 'a' (append).
    '''

    def __init__(self, file: str, mode: str = 'a') -> None:
        super().__init__()
        self.file = file
        self.mode = mode

    def consume(self, item: Any) -> None:
        '''
        Appends data to the file.

        Args:
            item: Data to append to the file. Should be serializable.
        '''
        with open(self.file, self.mode) as f:
            f.write(str(item))