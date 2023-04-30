from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import time

from ..core.node import ProducerNode


class IntProducer(ProducerNode):
    '''
    A ProducerNode that produces integers between start_value and end_value.
    '''

    def __init__(self, start_value: int = 0, end_value: int = None, wait_time_in_seconds: float = 0, fps=-1):
        '''
        Initializes the IntProducer.

        :param start_value: The starting value of the integer sequence. Default is 0.
        :param end_value: The ending value of the integer sequence. Default is None.
        :param wait_time_in_seconds: The number of seconds to wait between producing integers. Default is 0.
        :param fps: The number of frames per second to produce. Default is -1.
        '''
        self.start_value = start_value
        self.end_value = end_value
        self.wait_time_in_seconds = wait_time_in_seconds

        if fps > 0:
            self.wait_time_in_seconds = 1.0 / fps

        self.current_value = self.start_value
        super(IntProducer, self).__init__()

    def next(self):
        '''
        Returns the next integer in the sequence and waits for the specified number of seconds.

        :returns: The next integer in the sequence.
        '''
        if self.end_value is not None and self.current_value > self.end_value:
            raise StopIteration()

        integer_to_return = self.current_value
        self.current_value += 1
        time.sleep(self.wait_time_in_seconds)

        return integer_to_return
