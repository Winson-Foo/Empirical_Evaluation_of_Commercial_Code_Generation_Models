from typing import Any, List, Tuple
from pydispatch import dispatcher
from twisted.internet.defer import Deferred
from scrapy.utils import signal as scrapy_signal

class SignalManager:
    """
    A class to manage signals in Scrapy.
    """

    def __init__(self, sender: Any = dispatcher.Anonymous) -> None:
        """
        Initializes the SignalManager instance.

        :param sender: the default sender for signals.
        :type sender: object
        """
        self.sender = sender

    def connect(self, receiver: Any, signal: Any, **kwargs: Any) -> None:
        """
        Connects a receiver function to a signal.

        :param receiver: the function to be connected.
        :type receiver: collections.abc.Callable

        :param signal: the signal to connect to.
        :type signal: object

        :param kwargs: additional arguments to be passed to the signal.
        :type kwargs: dict
        """
        kwargs.setdefault("sender", self.sender)
        dispatcher.connect(receiver, signal, **kwargs)

    def disconnect(self, receiver: Any, signal: Any, **kwargs: Any) -> None:
        """
        Disconnects a receiver function from a signal.

        :param receiver: the function to be disconnected.
        :type receiver: collections.abc.Callable

        :param signal: the signal to disconnect from.
        :type signal: object

        :param kwargs: additional arguments to be passed to the signal.
        :type kwargs: dict
        """
        kwargs.setdefault("sender", self.sender)
        dispatcher.disconnect(receiver, signal, **kwargs)

    def send_and_log(self, signal: Any, **kwargs: Any) -> List[Tuple[Any, Any]]:
        """
        Sends a signal, catches exceptions and logs them.

        :param signal: the signal to send.
        :type signal: object

        :param kwargs: additional arguments to be passed to the signal.
        :type kwargs: dict

        :return: a list of tuples with the result of each receiver function.
        :rtype: list[tuple[object, object]]
        """
        kwargs.setdefault("sender", self.sender)
        return scrapy_signal.send_catch_log(signal, **kwargs)

    def send_deferred_and_log(self, signal: Any, **kwargs: Any) -> Deferred:
        """
        Sends a signal, catches exceptions and logs them.

        Returns a Deferred that gets fired once all signal handlers deferreds were fired.  

        :param signal: the signal to send.
        :type signal: object

        :param kwargs: additional arguments to be passed to the signal.
        :type kwargs: dict

        :return: a Deferred object with the results of the signal.
        :rtype: twisted.internet.defer.Deferred
        """
        kwargs.setdefault("sender", self.sender)
        return scrapy_signal.send_catch_log_deferred(signal, **kwargs)

    def disconnect_all(self, signal: Any, **kwargs: Any) -> None:
        """
        Disconnects all receivers from a signal.

        :param signal: the signal to disconnect from.
        :type signal: object

        :param kwargs: additional arguments to be passed to the signal.
        :type kwargs: dict
        """
        kwargs.setdefault("sender", self.sender)
        scrapy_signal.disconnect_all(signal, **kwargs) 