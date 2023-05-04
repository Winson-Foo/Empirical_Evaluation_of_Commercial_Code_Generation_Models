from typing import Any, List, Tuple
import scrapy.signals
from pydispatch import dispatcher
from twisted.internet.defer import Deferred

class SignalManager:
    def __init__(self, sender: Any = dispatcher.Anonymous):
        self.sender = sender

    def connect(self, receiver: Any, signal: Any, sender: Any = None) -> None:
        """
        Connect a receiver function to a signal.

        The signal can be any object, although Scrapy comes with some
        predefined signals that are documented in the :ref:`topics-signals`
        section.
        """
        if sender is None:
            sender = self.sender
        dispatcher.connect(receiver, signal, sender=sender)

    def disconnect(self, receiver: Any, signal: Any, sender: Any = None) -> None:
        """
        Disconnect a receiver function from a signal. This has the
        opposite effect of the `connect` method, and the arguments
        are the same.
        """
        if sender is None:
            sender = self.sender
        dispatcher.disconnect(receiver, signal, sender=sender)

    def send_catch_log(self, signal: Any, sender: Any = None, **signal_kwargs: Any) -> List[Tuple[Any, Any]]:
        """
        Send a signal, catch exceptions and log them.

        The keyword arguments are passed to the signal handlers (connected
        through the `connect` method).
        """
        if sender is None:
            sender = self.sender
        return scrapy.signals.send_catch_log(signal, sender=sender, **signal_kwargs)

    def send_catch_log_deferred(self, signal: Any, sender: Any = None, **signal_kwargs: Any) -> Deferred:
        """
        Like `send_catch_log` but supports returning
        `Deferred` objects from signal handlers.

        Returns a Deferred that gets fired once all signal handlers
        deferreds were fired. Send a signal, catch exceptions and log them.

        The keyword arguments are passed to the signal handlers (connected
        through the `connect` method).
        """
        if sender is None:
            sender = self.sender
        return scrapy.signals.send_catch_log_deferred(signal, sender=sender, **signal_kwargs)

    def disconnect_all(self, signal: Any, sender: Any = None) -> None:
        """
        Disconnect all receivers from the given signal.
        """
        if sender is None:
            sender = self.sender
        scrapy.signals.disconnect_all(signal, sender=sender) 