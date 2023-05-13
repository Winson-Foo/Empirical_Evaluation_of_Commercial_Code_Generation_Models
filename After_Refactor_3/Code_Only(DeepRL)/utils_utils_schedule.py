class ConstantSchedule:
    def __init__(self, val):
        self.val = val

    def __call__(self, steps=1):
        return self.val


class LinearSchedule:
    def __init__(self, start, end=None, steps=None):
        self.start = start
        self.end = end
        self.steps = steps

    def __call__(self, steps=1):
        if self.end is None:
            self.end = self.start
            self.steps = 1
        inc = (self.end - self.start) / float(self.steps)
        current = self.start
        if self.end > self.start:
            bound = min
        else:
            bound = max
        val = current
        current = bound(current + inc * steps, self.end)
        return val