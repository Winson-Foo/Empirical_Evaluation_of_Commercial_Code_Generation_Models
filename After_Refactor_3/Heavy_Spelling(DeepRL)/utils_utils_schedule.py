class ConstantSchedule:
    def __init__(self, val):
        self.val = val

    def __call__(self, steps=1):
        return self.val


class LinearSchedule:
    def __init__(self, start, end=None, steps=None):
        if end is None:
            end = start
            steps = 1

        self.start = start
        self.end = end
        self.steps = steps

        # Calculate step size
        self.step_size = (end - start) / float(steps)

        # Set initial value
        self.current = start

        # Determine bound function to use
        self.bound_func = min if end > start else max

    def __call__(self, steps=1):
        # Get current value
        val = self.current

        # Calculate new value, bounded by start and end
        self.current = self.bound_func(
            self.current + self.step_size * steps, self.start, self.end
        )

        return val

    def __repr__(self):
        return f"LinearSchedule(start={self.start}, end={self.end}, steps={self.steps})"

    def __str__(self):
        return f"LinearSchedule from {self.start} to {self.end} in {self.steps} steps" 