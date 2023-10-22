class Schedule:
    def __init__(self, start, end=None, steps=None):
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
        pass

class ConstantSchedule(Schedule):
    def __init__(self, val):
        self.val = val
        super().__init__(val, val)

    def __call__(self, steps=1):
        return self.val

class LinearSchedule(Schedule):
    def __init__(self, start, end=None, steps=None):
        super().__init__(start, end, steps)

    def __call__(self, steps=1):
        val = self.current
        self.current = self.bound(self.current + self.inc * steps, self.end)
        return val
