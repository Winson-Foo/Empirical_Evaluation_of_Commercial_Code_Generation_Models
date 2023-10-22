class ConstantSchedule:
    """
    A constant schedule that returns the same value all the time.

    Args:
        val (float): the constant value to be returned
    """

    def __init__(self, val):
        self.val = val

    def __call__(self, steps=1):
        """
        Returns the constant value.

        Args:
            steps (int, optional): the number of steps taken so far

        Returns:
            float: the constant value
        """

        return self.val


class LinearSchedule:
    """
    A linear schedule that interpolates between a start and end value
    linearly over a given number of steps.

    Args:
        start (float): the start value
        end (float, optional): the end value (default: start)
        steps (int, optional): the number of steps to interpolate over (default: 1)
    """

    def __init__(self, start, end=None, steps=1):
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
        Returns the interpolated value for the current step and updates
        the internal state for the next step.

        Args:
            steps (int, optional): the number of steps taken so far

        Returns:
            float: the interpolated value for the current step
        """

        val = self.current
        self.current = self.bound(self.current + self.inc * steps, self.end)
        return val