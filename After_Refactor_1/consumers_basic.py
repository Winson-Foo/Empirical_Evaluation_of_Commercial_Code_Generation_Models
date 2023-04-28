from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import json
import requests
from ..core.node import ConsumerNode


class CommandlineConsumer(ConsumerNode):
    '''
    Writes the input received to the command line.

    - Arguments:
        - sep: separator to use between tokens.
        - end: end of line character
    '''
    def __init__(self, sep=' ', end='\n'):
        super(CommandlineConsumer, self).__init__()
        self._end = end
        self._sep = sep
    
    def consume(self, item):
        '''
        Prints `item` to the command line, adding an end of line character after it.

        - Arguments:
            - item: It can be anything that can be printed with the ``print()`` function
        '''
        print(item, sep=self._sep, end=self._end)


class VoidConsumer(ConsumerNode):
    '''
    Ignores the input received.
    Helpful in debugging flows.
    '''
    def __init__(self):
        super(VoidConsumer, self).__init__()
    
    def consume(self, item):
        '''
        Does nothing with the item passed
        '''
        pass


class WebhookConsumer(ConsumerNode):
    '''
    Sends a JSON payload as a request to a specified host.

    - Arguments:
        - host: Address of the host.
        - method: HTTP method to use, defaults to POST.
    '''
    def __init__(self, host, method="POST"):
        super(WebhookConsumer, self).__init__()
        self.host = host
        self.method = method

    def consume(self, item):
        '''
        Converts `item` to a JSON payload and sends it as a request to the specified host.
        '''
        try:
            payload = json.dumps(item)
        except TypeError:
            print("Not consuming item is not JSON serializable")
            return

        response = requests.request(self.method, self.host, json=payload)
        response.raise_for_status()


class FileAppenderConsumer(ConsumerNode):
    '''
    Appends the input received to a specified file.

    - Arguments:
        - filepath: Path of the file to append to.
    '''
    def __init__(self, filepath):
        super(FileAppenderConsumer, self).__init__()
        self.filepath = filepath

    def consume(self, item):
        '''
        Appends `item` to the specified file.

        Raises:
            NotImplementedError: If called, since the behavior of this class is not implemented.
        '''
        raise NotImplementedError("FileAppenderConsumer not yet implemented")