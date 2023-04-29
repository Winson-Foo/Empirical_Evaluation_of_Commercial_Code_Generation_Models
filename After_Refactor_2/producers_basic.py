import time
from typing import Optional
from ..core.node import ProducerNode

class IntProducer(ProducerNode):
    """
    Each time the `next` method is called, produces the next
    integer in the sequence between `start_value` and `end_value`.
    
    If `wait_time` is greater than zero, it sleeps for the specified
    amount of seconds each time `next` is called.

    If `fps` is given a value greater than 0, it overrides the value
    of `wait_time`.
    """
    def __init__(self, start_value: int = 0, end_value: Optional[int] = None, 
                wait_time: float = 0, fps: int = -1) -> None:
        self.start_value = start_value
        self.end_value = end_value
        self.wait_time = wait_time if fps < 0 else 1.0 / fps
        self.current_value = self.start_value
        super(IntProducer, self).__init__()

    def next(self) -> int:
        """
        Returns the next integer in the sequence between `start_value`
        and `end_value`.
        
        Raises `StopIteration` if there are no more integers to produce.
        """
        if self.end_value is not None and self.current_value > self.end_value:
            raise StopIteration
        to_return = self.current_value
        self.current_value += 1
        time.sleep(self.wait_time)
        return to_return