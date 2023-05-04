from typing import Any, List, Tuple

# We can use the built-in `signal` module instead of the `pydispatch` library.
import signal
from twisted.internet.defer import Deferred
from scrapy.utils import signal as scrapy_signal


class SignalManager:
    def __init__(self, sender: Any = signal.SIGUSR1):
        """
        Initialize SignalManager object.

        :param sender: the sender of the signal
        :type sender: Any
        """
        self.sender = sender

    def connect(self, receiver: Any, signal_name: str, **kwargs: Any) -> None:
        """
        Connect a receiver function to a signal.

        The signal can be any object, although Scrapy comes with some
        predefined signals that are documented in the :ref:`topics-signals`
        section.

        :param receiver: the function to be connected
        :type receiver: collections.abc.Callable

        :param signal_name: the name of the signal to connect to
        :type signal_name: str
        """
        kwargs.setdefault("sender", self.sender)
        signal.signal(signal_name, receiver, **kwargs)

    def disconnect(self, receiver: Any, signal_name: str, **kwargs: Any) -> None:
        """
        Disconnect a receiver function from a signal. This has the
        opposite effect of the :meth:`connect` method, and the arguments
        are the same.

        :param receiver: the function to be disconnected
        :type receiver: collections.abc.Callable

        :param signal_name: the name of the signal to disconnect from
        :type signal_name: str
        """
        kwargs.setdefault("sender", self.sender)
        signal.signal(signal_name, None, **kwargs)

    def send_catch_log(self, signal_name: str, **kwargs: Any) -> List[Tuple[Any, Any]]:
        """
        Send a signal, catch exceptions and log them.

        The keyword arguments are passed to the signal handlers (connected
        through the :meth:`connect` method).

        :param signal_name: the name of the signal to send
        :type signal_name: str

        :return: a list of tuples containing the receiver function and its return value
        :rtype: List[Tuple[Any, Any]]
        """
        kwargs.setdefault("sender", self.sender)
        return scrapy_signal.send_catch_log(signal_name, **kwargs)

    def send_catch_log_deferred(self, signal_name: str, **kwargs: Any) -> Deferred:
        """
        Like :meth:`send_catch_log` but supports returning
        :class:`~twisted.internet.defer.Deferred` objects from signal handlers.

        Returns a Deferred that gets fired once all signal handlers
        deferreds were fired. Send a signal, catch exceptions and log them.

        The keyword arguments are passed to the signal handlers (connected
        through the :meth:`connect` method).

        :param signal_name: the name of the signal to send
        :type signal_name: str

        :return: a Deferred that gets fired once all signal handlers deferreds were fired
        :rtype: Deferred
        """
        kwargs.setdefault("sender", self.sender)
        return scrapy_signal.send_catch_log_deferred(signal_name, **kwargs)

    def disconnect_all(self, signal_name: str, **kwargs: Any) -> None:
        """
        Disconnect all receivers from the given signal.

        :param signal_name: the name of the signal to disconnect from
        :type signal_name: str
        """
        kwargs.setdefault("sender", self.sender)
        scrapy_signal.disconnect_all(signal_name, **kwargs) 