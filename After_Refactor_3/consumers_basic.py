import json
import requests
from ..core.node import ConsumerNode


class CommandlineConsumer(ConsumerNode):
    """
    Writes the input received to the command line.

    Args:
        sep (str): separator to use between tokens.
        end (str): end of line character
    """
    def __init__(self, sep=' ', end='\n'):
        super(CommandlineConsumer, self).__init__()
        self.end_of_line = end
        self.separator = sep

    def consume(self, item):
        """
        Prints `item` to the command line, adding an end of line character after it.

        Args:
            item: It can be anything that can be printed with the `print()` function
        """
        print(item, sep=self.separator, end=self.end_of_line)


class VoidConsumer(ConsumerNode):
    """
    Ignores the input received.
    Helpful in debugging flows.
    """
    def __init__(self):
        super(VoidConsumer, self).__init__()

    def consume(self, item):
        """
        Does nothing with the item passed
        """
        pass


class WebhookConsumer(ConsumerNode):
    """
    Sends data to webhook URL as JSON Object.

    Args:
        host (str): webhook url.
        method (str): HTTP request method.

    """
    def __init__(self, host, method="post"):
        super(WebhookConsumer, self).__init__()
        self.host = host
        self.method = method

    def consume(self, item):
        """
        Sends data to webhook URL as JSON Object.

        Args:
            item: It can be anything that can be converted to a JSON Object.

        """
        try:
            data = json.loads(item)
        except TypeError:
            print("Not consuming item is not json serializable")
        response = requests.post(self.host, json=data)
        return response


class FileAppenderConsumer(ConsumerNode):
    """
    Appends the data to the file.

    """
    def __init__(self, file_path):
        super(FileAppenderConsumer, self).__init__()
        self.file_path = file_path

    def consume(self, item):
        """
        Appends the data to the file.

        Args:
            item: It can be anything that can be serialized.

        """
        with open(self.file_path, mode='a') as file:
            file.write(item)