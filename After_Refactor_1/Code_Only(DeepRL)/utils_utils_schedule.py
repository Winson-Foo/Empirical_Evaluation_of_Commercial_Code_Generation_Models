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
        self.set_increment()

    def set_increment(self):
        self.inc = (self.end - self.start) / float(self.steps)

    def get_value(self):
        return self.current

    def update(self, steps=1):
        self.current = self.bound(self.current + self.inc * steps, self.end)
        return self.get_value()

    def __call__(self, steps=1):
        return self.get_value()


class IncreasingLinearSchedule(LinearSchedule):
    def __init__(self, start, end, steps=None):
        super().__init__(start, end, steps)
        self.bound = min


class DecreasingLinearSchedule(LinearSchedule):
    def __init__(self, start, end, steps=None):
        super().__init__(start, end, steps)
        self.bound = max