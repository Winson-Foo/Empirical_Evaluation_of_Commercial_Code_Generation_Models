class ConstantSchedule:
    """
    Represents a constant schedule that always returns the same value
    """
    def __init__(self, val):
        self.val = val

    def __call__(self, steps=1):
        return self.val


class LinearSchedule:
    """
    Represents a linear schedule that gradually increases or decreases its value
    """
    def __init__(self, start, end=None, steps=None):
        """
        Initializes the linear schedule with a starting value, an ending value (optional), and a number of steps (optional)
        """
        if end is None:
            end = start
            steps = 1
        self.inc = (end - start) / float(steps)
        self.current = start
        self.end = end
        self.bound = min if end > start else max

    def __call__(self, steps=1):
        """
        Updates the current value of the linear schedule based on the number of steps taken and returns the previous value
        """
        val = self.current
        self.current = self.bound(self.current + self.inc * steps, self.end)
        return val