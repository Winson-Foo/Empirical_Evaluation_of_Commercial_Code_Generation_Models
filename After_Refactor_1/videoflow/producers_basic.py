class IntProducer(ProducerNode):
    '''
    Each time the ``next`` method is called, produces the next
    integer in the sequence between ``start_value`` and ``end_value``.
    
    If ``wait_time_in_seconds`` is greater than zero, it sleeps for 
    the specified amount of seconds each time ``next()`` is called.

    If ``fps`` is given a value greater than 0, ``fps`` overrides the 
    value of ``wait_time_in_seconds``
    '''

    def __init__(self, start_value: int = 0, end_value: int = None, wait_time: float = 0, fps: int = -1):
        # Use more descriptive variable names
        self.start_value = start_value
        self.end_value = end_value
        self.wait_time = wait_time
        self.fps = fps
        self.current_value = start_value

        # Simplify the logic for setting the wait time
        if self.fps > 0:
            self.wait_time = 1.0 / self.fps

        # super() doesn't need any arguments in Python 3
        super().__init__()

    def next(self):
        '''
        - Returns:
            - next: an integer
        '''
        if self.end_value is not None and self.current_value > self.end_value:
            raise StopIteration()

        to_return = self.current_value
        self.current_value += 1

        time.sleep(self.wait_time)

        return to_return
