# Refactored code:

class Schedule:
    def __init__(self, val):
        self.val = val

    def __call__(self, steps=1):
        return self.val

class LinearSchedule(Schedule):
    def __init__(self, start, end=None, steps=None):
        super().__init__(val=None)
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
        if self.val is None:
            self.val = self.current
        else:
            self.val = self.bound(self.current + self.inc * steps, self.end)
        return self.val

class ConstantSchedule(Schedule):
    def __init__(self, val):
        super().__init__(val=val)