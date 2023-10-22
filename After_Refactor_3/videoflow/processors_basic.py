from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import time

from ..core.node import ProcessorNode


class DelayProcessor(ProcessorNode):
    '''
    DelayProcessor implements a delay
    function: it introduces delay by setting fps 
    to a value greater than 0.

    - Arguments:
        - fps (int): frames per second. If value is less \
            than or equal to zero, it is ignored, and no delay \
            is introduced.
    '''
    def __init__(self, fps=-1, *args, **kargs):
        super(DelayProcessor, self).__init__(*args, **kargs)
        self._fps = fps

    def process(self, inp):
        if self._fps > 0:
            wait_time = 1.0 / self._fps
            time.sleep(wait_time)
        return inp


class IdentityProcessor(DelayProcessor):
    '''
    IdentityProcessor implements the identity
    function: it returns the same value that it received
    as input. 

    - Arguments:
        - fps (int): frames per second. If value is less \
            than or equal to zero, it is ignored, and no delay \
            is introduced.
    '''
    def __init__(self, fps=-1, *args, **kargs):
        super(IdentityProcessor, self).__init__(fps, *args, **kargs)


class JoinerProcessor(DelayProcessor):
    '''
    JoinerProcessor takes all the parameters received in the 
    ``process`` method and makes them a tuple of items.

    - Arguments:
        - fps (int): frames per second. If value is less \
            than or equal to zero, it is ignored, and no delay \
            is introduced.
    '''
    def __init__(self, fps=-1, *args, **kargs):
        super(JoinerProcessor, self).__init__(fps, *args, **kargs)
    
    def process(self, *inp):
        if self._fps > 0:
            wait_time = 1.0 / self._fps
            time.sleep(wait_time)
        return tuple(inp)