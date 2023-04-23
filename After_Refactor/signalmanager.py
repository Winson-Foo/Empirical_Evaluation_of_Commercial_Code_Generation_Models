from typing import Any, List, Tuple
from pydispatch import dispatcher
from twisted.internet.defer import Deferred
from scrapy.utils import signal as _signal

class SignalManager:
    def __init__(self, sender: Any = dispatcher.Anonymous):
        self.sender = sender

    def connect(self, receiver: Any, signal: Any, **kwargs: Any) -> None:
        kwargs.setdefault("sender", self.sender)
        dispatcher.connect(receiver, signal, **kwargs)

    def disconnect(self, receiver: Any, signal: Any, **kwargs: Any) -> None:
        kwargs.setdefault("sender", self.sender)
        dispatcher.disconnect(receiver, signal, **kwargs)

    def send_catch_log(self, signal: Any, **kwargs: Any) -> List[Tuple[Any, Any]]:
        kwargs.setdefault("sender", self.sender)
        return _signal.send_catch_log(signal, **kwargs)

    def send_catch_log_deferred(self, signal: Any, **kwargs: Any) -> Deferred:
        kwargs.setdefault("sender", self.sender)
        return _signal.send_catch_log_deferred(signal, **kwargs)

    @staticmethod
    def disconnect_all(signal: Any, **kwargs: Any) -> None:
        kwargs.setdefault("sender", dispatcher.Anonymous)
        _signal.disconnect_all(signal, **kwargs)

    def set_default_sender(self, sender: Any):
        self.sender = sender

    def reset_sender(self):
        self.sender = dispatcher.Anonymous