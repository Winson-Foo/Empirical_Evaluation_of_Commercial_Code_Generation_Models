from typing import Any, Tuple
import time
from ..core.node import ProcessorNode

class IdentityProcessor(ProcessorNode):
    '''
    IdentityProcessor implements the identity
    function: it returns the same value that it received
    as input. You can introduce some delay by setting frames_per_second 
    to a value greater than 0.

    - Arguments:
        - frames_per_second (int): frames per second. If value is less \
            than or equal to zero, it is ignored, and no delay \
            is introduced.
    '''
    def __init__(self, frames_per_second: int = 0, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.frames_per_second = frames_per_second

    def process(self, inp: Any) -> Any:
        if self.frames_per_second > 0:
            time.sleep(1 / self.frames_per_second)
        return inp

class JoinerProcessor(ProcessorNode):
    '''
    Takes all the parameters received in the ``process`` method
    and makes them a tuple of items.

    - Arguments:
        - frames_per_second (int): frames per second. If value is less \
            than or equal to zero, it is ignored, and no delay \
            is introduced.
    '''
    def __init__(self, frames_per_second: int = 0, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.frames_per_second = frames_per_second

    def process(self, *inp: Any) -> Tuple[Any, ...]:
        if self.frames_per_second > 0:
            time.sleep(1 / self.frames_per_second)
        return tuple(inp)
