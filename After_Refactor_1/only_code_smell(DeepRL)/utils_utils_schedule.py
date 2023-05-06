class ConstantSchedule:
    def __init__(self, val):
        self.val = val

    def __call__(self, steps=1):
        return self.val


class LinearSchedule:
    def __init__(self, start, end=None, steps=None):
        self.start = start

        if end is None:
            end = start
            steps = 1

        self.end = end
        self.steps = steps
        self.inc = (end - start) / float(steps)
        self.current = start
        self.bound = min if end > start else max

    def __call__(self, steps=1):
        val = self.current
        self.current = self.bound(self.current + self.inc * steps, self.end)
        return val