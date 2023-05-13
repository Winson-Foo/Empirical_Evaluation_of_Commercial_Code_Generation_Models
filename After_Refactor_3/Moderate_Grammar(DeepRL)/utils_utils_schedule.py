class Schedule:
    def __call__(self, steps: int = 1) -> float:
        raise NotImplementedError


class ConstantSchedule(Schedule):
    def __init__(self, value: float):
        self.value = value

    def __call__(self, steps: int = 1) -> float:
        return self.value


class LinearSchedule(Schedule):
    def __init__(self, start: float, end: Optional[float] = None, steps: Optional[int] = None):
        if end is None:
            end = start
            steps = 1
        self.inc = (end - start) / float(steps)
        self.current = start
        self.end = end

    def __call__(self, steps: int = 1) -> float:
        val = self.current
        self.current = max(min(self.current + self.inc * steps, self.end), self.current)
        return val

    @property
    def bound(self):
        return min if self.end > self.current else max