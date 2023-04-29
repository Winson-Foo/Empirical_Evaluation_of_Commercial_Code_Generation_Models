class AggregatorInterface:
    '''
    Aggregator Interface defines the basic behavior necessary
    for any aggregator to maintain results over a stream.
    '''
    def process(self, inp):
        '''
        Updates internal aggregates with the new 
        input value and returns the updated aggregate
        '''
        raise NotImplementedError

class SumAggregator(AggregatorInterface):
    '''
    Keeps a running sum of all the inputs processed
    '''
    def __init__(self):
        self._sum = 0
    
    def process(self, inp):
        '''
        - Arguments:
            - inp: a number
        
        - Returns:
            - sum: the cumulative sum up to this point, including ``inp`` in it.
        '''
        self._sum += inp
        return self._sum

class MultiplicationAggregator(AggregatorInterface):
    '''
    Keeps a running multiplication of all the inputs processed
    '''
    def __init__(self):
        self._mult = 1

    def process(self, inp):
        '''
        - Arguments:
            - inp: a number
        
        - Returns:
            - mult: the cumulative multiplication up to this point, including ``inp`` in it.
        '''
        self._mult *= inp
        return self._mult  

class CountAggregator(AggregatorInterface):
    '''
    Keeps count of all the items processed
    '''
    def __init__(self):
        self._count = 0
    
    def process(self, inp):
        '''
        - Arguments:
            - inp: a number
        
        - Returns:
            - count: the cumulative count up to this point
        '''
        self._count += 1
        return self._count

class MaxAggregator(AggregatorInterface):
    '''
    Keeps track of the largest number seen so far
    '''
    def __init__(self):
        self._max = float("-inf")
    
    def process(self, inp):
        '''
        - Arguments:
            - inp: a number
        
        - Returns:
            - max: the maximum seen value up to this point
        '''
        if inp > self._max:
            self._max = inp
        return self._max

class MinAggregator(AggregatorInterface):
    '''
    Keeps track of the smallest number seen so far
    '''
    def __init__(self):
        self._min = float("inf")
    
    def process(self, inp):
        '''
        - Arguments:
            - inp: a number
        
        - Returns:
            - min: the minimum seen value up to this point
        '''
        if inp < self._min:
            self._min = inp
        return self._min
