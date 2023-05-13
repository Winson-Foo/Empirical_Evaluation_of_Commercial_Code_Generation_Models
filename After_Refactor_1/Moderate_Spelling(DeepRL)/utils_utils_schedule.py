class ConstantSchedule:
    """
    Constant learning rate scheduler that returns a predetermined value
    """

    def __init__(self, value):
        self.value = value

    def __call__(self, steps=1):
        """
        Returns the preset learning rate value
        """
        return self.value

class LinearSchedule:
    """
    Linear learning rate scheduler that gradually decreases the learning rate
    from a starting value to an ending value over a specified number of steps
    """

    def __init__(self, start_value, end_value=None, num_steps=None):
        """
        Initializes the scheduler with a starting value, ending value, and
        number of steps across which the learning rate is decreased
        """
        if end_value is None:
            end_value = start_value
            num_steps = 1
        self.increment = (end_value - start_value) / float(num_steps)
        self.current_value = start_value
        self.end_value = end_value
        self._set_bound()

    def _set_bound(self):
        """
        Sets the bound function depending on whether the learning rate is
        increasing or decreasing
        """
        if self.end_value > self.current_value:
            self.bound = min
        else:
            self.bound = max

    def _get_next_value(self, steps):
        """
        Calculates the next learning rate value after a specified number of steps
        """
        next_value = self.current_value + self.increment * steps
        return self.bound(next_value, self.end_value)

    def __call__(self, steps=1):
        """
        Returns the current learning rate and updates it to the next value
        """
        current_value = self.current_value
        self.current_value = self._get_next_value(steps)
        return current_value