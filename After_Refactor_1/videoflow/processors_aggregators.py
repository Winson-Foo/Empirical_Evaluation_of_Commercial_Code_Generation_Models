class SumAggregator:
    """Keeps a running sum of all the inputs processed."""
    def __init__(self):
        self._sum = 0
    
    def process(self, num):
        """Add num to the sum and return the updated sum."""
        self._sum += num
        return self._sum

class MultiplicationAggregator:
    """Keeps a running multiplication of all the inputs processed."""
    def __init__(self):
        self._product = 1

    def process(self, num):
        """Multiply num to the product and return the updated product."""
        self._product *= num
        return self._product  

class CountAggregator:
    """Keeps count of all the items processed."""
    def __init__(self):
        self._count = 0

    def process(self, item):
        """Increment the count and return the updated count."""
        self._count += 1
        return self._count

class MaxAggregator:
    """Keeps track of the maximum value seen."""
    def __init__(self):
        self._max = float("-inf")

    def process(self, num):
        """
        Update the maximum value and return the updated maximum value.
        Args:
        - num: a number
        """
        if num > self._max:
            self._max = num
        return self._max

class MinAggregator:
    """Keeps track of the minimum value seen."""
    def __init__(self):
        self._min = float("inf")

    def process(self, num):
        """
        Update the minimum value and return the updated minimum value.
        Args:
        - num: a number
        """
        if num < self._min:
            self._min = num
        return self._min