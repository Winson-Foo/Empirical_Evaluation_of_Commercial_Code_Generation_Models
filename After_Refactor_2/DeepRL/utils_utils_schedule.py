class ConstantSchedule:
    def __init__(self, val):
        self.val = val

    def __call__(self, steps=1):
        return self.val


class LinearSchedule:
    def __init__(self, start, end=None, steps=None):
        self.start = start
        self.end = end or start
        self.steps = steps or 1
        self.inc = (self.end - self.start) / float(self.steps)
        self.current = self.start
        self.bound = min if self.end > self.start else max

    def __call__(self, steps=1):
        val = self.current
        self.current = self.bound(self.current + self.inc * steps, self.end)
        return val