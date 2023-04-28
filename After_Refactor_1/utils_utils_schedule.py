class ConstantSchedule:
    def __init__(self, value):
        self.value = value

    def __call__(self, num_steps=1):
        return self.value


class LinearSchedule:
    def __init__(self, start_value, end_value=None, num_steps=None):
        if end_value is None:
            end_value = start_value
            num_steps = 1
        self.start_value = start_value
        self.end_value = end_value
        self.num_steps = num_steps
        self.increment = (end_value - start_value) / float(num_steps) if num_steps > 0 else 0
        self.current_value = start_value

    def __call__(self, num_steps=1):
        value = self.current_value
        self.current_value = self.bound(self.current_value + self.increment * num_steps, self.end_value)
        return value

    def bound(self, value, bound_value):
        return min(max(value, self.start_value), bound_value) if self.end_value > self.start_value else max(min(value, self.start_value), bound_value)