class ConstantSchedule:
    """
    A class that represents a constant schedule.
    """
    def __init__(self, val):
        """
        Constructor for the ConstantSchedule class.

        :param val: (float) The value that should be returned for every __call__ invocation.
        """
        self.val = val

    def __call__(self, steps=1):
        """
        Returns the constant value specified by the `val` parameter of the constructor.

        :param steps: (int) The number of steps that the caller has taken. (not used)
        :return: The value specified by the `val` parameter of the constructor.
        """
        return self.val


class LinearSchedule:
    """
    A class that represents a linear schedule. The returned value is incremented every time the __call__ method is invoked.
    """
    def __init__(self, start, end=None, steps=None):
        """
        Constructor for the LinearSchedule class.

        :param start: (float) The value that should be returned for the first __call__ invocation.
        :param end: (float) The value that should be returned after `steps` steps have been taken.
        :param steps: (int) The number of steps after which `end` should be returned. If None, `end` is never returned.
        """
        if end is None:
            end = start
            steps = 1
        self.inc = (end - start) / float(steps)
        self.current = start
        self.end = end
        if end > start:
            self.bound = min
        else:
            self.bound = max

    def __call__(self, steps=1):
        """
        Returns the current value specified by the instance variables. Updates the current value.
        If `steps` is not 1, updates the current value `steps` times.

        :param steps: (int) The number of steps that the caller has taken.
        :return: (float) The current value specified by the instance variables.
        """
        val = self.current
        self.current = self.bound(self.current + self.inc * steps, self.end)
        return val