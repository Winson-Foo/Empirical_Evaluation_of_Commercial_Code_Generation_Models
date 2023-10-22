class ConstantSchedule:
    def __init__(self, constant_value):
        """
        :param constant_value: The value that the schedule should return every time it is called
        """
        self.constant_value = constant_value

    def __call__(self, steps=1):
        """
        :param steps: number of steps taken
        :return: The constant value
        """
        return self.constant_value


class LinearSchedule:
    def __init__(self, start, end=None, steps=None):
        """
        :param start: The starting value of the schedule
        :param end: The end value of the schedule
        :param steps: The number of steps the schedule should take to go from start to end
        """
        if end is None:
            end = start
            steps = 1
        self.increment = (end - start) / float(steps)
        self.current_value = start
        self.end_value = end
        self.bound_function = min if end > start else max

    def __call__(self, steps=1):
        """
        :param steps: The number of steps taken
        :return: The current value of the schedule
        """
        value = self.current_value
        self.current_value = self.bound_function(self.current_value + self.increment * steps, self.end_value)
        return value