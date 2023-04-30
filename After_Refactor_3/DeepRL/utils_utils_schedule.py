class ConstantScheduler:
    """A scheduler that always returns a constant value"""
    def __init__(self, value: float):
        self.value = value

    def step(self, steps: int = 1) -> float:
        """Return the constant value"""
        return self.value


class LinearScheduler:
    """A scheduler that linearly increases or decreases a value from a start value to an end value"""
    def __init__(self, start: float, end: float = None, num_steps: int = None):
        """
        Initialize a linear scheduler.

        Args:
            start: The start value of the scheduler
            end: The end value of the scheduler (default: same as start)
            num_steps: The total number of steps in the scheduler (default: 1)
        """
        self.start = start
        self.end = end if end is not None else start
        self.num_steps = num_steps if num_steps is not None else 1
        self.increment = (self.end - self.start) / float(self.num_steps)
        self.current = self.start

    def step(self, steps: int = 1) -> float:
        """
        Compute the next value in the schedule.

        Args:
            steps: The number of steps to take (default: 1)

        Returns:
            The current value in the schedule
        """
        current_value = self.current
        self.current = min(self.current + self.increment * steps, self.end) if self.end > self.start else max(
            self.current - self.increment * steps, self.end)
        return current_value

# Named constants
CONSTANT_SCHEDULE_VALUE = 0.5
LINEAR_SCHEDULE_START = 1.0
LINEAR_SCHEDULE_END = 0.1
LINEAR_SCHEDULE_NUM_STEPS = 100000

# Example usage
constant_scheduler = ConstantScheduler(CONSTANT_SCHEDULE_VALUE)
linear_scheduler = LinearScheduler(start=LINEAR_SCHEDULE_START, end=LINEAR_SCHEDULE_END,
                                   num_steps=LINEAR_SCHEDULE_NUM_STEPS)

for i in range(10):
    print(f"Step {i}: Constant value = {constant_scheduler.step()}, Linear value = {linear_scheduler.step()}")