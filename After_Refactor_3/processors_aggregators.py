from typing import List

from ..core.node import OneTaskProcessorNode

class Aggregator(OneTaskProcessorNode):
    """
    A base class for all of the aggregator nodes.
    """
    def __init__(self):
        super().__init__()
    
    def process(self, inp: float) -> float:
        """
        Process a single input value.
        
        Arguments:
        - inp: The input value.
        
        Returns:
        - The result of aggregating the input value with the previous inputs.
        """
        raise NotImplementedError

class SumAggregator(Aggregator):
    '''
    Keeps a running sum of all the inputs processed
    '''
    def __init__(self):
        super().__init__()
        self._sum = 0
    
    def process(self, inp: float) -> float:
        '''
        Add the input value to the current sum, and return the new sum.
        '''
        self._sum += inp
        return self._sum

class MultiplicationAggregator(Aggregator):
    '''
    Keeps a running multiplication of all the inputs processed
    '''
    def __init__(self):
        super().__init__()
        self._mult = 1

    def process(self, inp: float) -> float:
        '''
        Multiply the input value by the current value, and return the new product.
        '''
        self._mult *= inp
        return self._mult  

class CountAggregator(Aggregator):
    '''
    Keeps count of all the items processed
    '''
    def __init__(self):
        super().__init__()
        self._count = 0
    
    def process(self, inp: float) -> float:
        '''
        Increment the count by 1, and return the new count.
        '''
        self._count += 1
        return self._count

class MaxAggregator(Aggregator):
    '''
    Keeps track of the maximum value seen so far
    '''
    def __init__(self):
        super().__init__()
        self._max = float("-inf")
    
    def process(self, inp: float) -> float:
        '''
        Update the maximum value seen if the input value is greater than the current maximum, and return the new maximum.
        '''
        if inp > self._max:
            self._max = inp
        return self._max

class MinAggregator(Aggregator):
    '''
    Keeps track of the minimum value seen so far
    '''
    def __init__(self):
        super().__init__()
        self._min = float("inf")
    
    def process(self, inp: float) -> float:
        '''
        Update the minimum value seen if the input value is less than the current minimum, and return the new minimum.
        '''
        if inp < self._min:
            self._min = inp
        return self._min