import time
from typing import Tuple

from ..core.node import ProcessorNode


class IdentityProcessor(ProcessorNode):
    """
    Implements the identity function: it returns the same value that it received
    as input. You can introduce some delay by setting fps to a value greater than 0.

    Args:
        fps (int): Frames per second. If value is less than or equal to zero, 
            it is ignored, and no delay is introduced.
    """
    def __init__(self, fps: int = -1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wait_time = 1.0 / fps if fps > 0 else 0

    def process(self, input_data):
        """
        Process the input data.

        Args:
            input_data: Data to be processed.

        Returns:
            The same input data.
        """
        if self.wait_time > 0:
            time.sleep(self.wait_time)
        return input_data


class JoinerProcessor(ProcessorNode):
    """
    Takes all the parameters received in the process method and makes them 
    a tuple of items.

    Args:
        fps (int): Frames per second. If value is less than or equal to zero, 
            it is ignored, and no delay is introduced.
    """
    def __init__(self, fps: int = -1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wait_time = 1.0 / fps if fps > 0 else 0
    
    def process(self, *inputs) -> Tuple:
        """
        Process multiple input data.

        Args:
            *inputs: Multiple data to be processed.

        Returns:
            Tuple of all input data.
        """
        if self.wait_time > 0:
            time.sleep(self.wait_time)
        return tuple(inputs)