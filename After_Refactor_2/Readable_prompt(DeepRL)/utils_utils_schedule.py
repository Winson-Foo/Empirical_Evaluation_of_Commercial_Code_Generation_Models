# Refactored Code

# Constant Schedule Function
class ConstantSchedule:
    def __init__(self, val):
        self.val = val

    def __call__(self, steps=1):
        return self.val


# Linear Schedule Function
class LinearSchedule:
    def __init__(self, start, end=None, steps=None):
        '''
        Takes a start value and either an end value or steps value to compute final value
        '''
        if end is None:
            end = start
            steps = 1
        self.inc = (end - start) / float(steps) # Increment value computed
        self.current = start
        self.end = end
        self.bound = min if end > start else max # Bound set based on relative values of start and end input

    def __call__(self, steps=1):
        '''
        :param steps: number of steps to take to reach the final value
        :return: Current linearly interpolated value
        '''
        val = self.current
        self.current = self.bound(self.current + self.inc * steps, self.end) # Increment the current value
        return val # Return the current interpolated value