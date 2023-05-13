class ConstantSchedule:
    """
    A class used to represent a constant schedule.

    ...

    Attributes
    ----------
    val : float
        value of the schedule

    Methods
    -------
    __call__(steps=1)
        Returns the constant value for any given step
    """

    def __init__(self, val):
        """
        Constructs all the necessary attributes for the constant schedule object.

        Parameters
        ----------
            val : float
                Value of the schedule
        """
        self.val = val

    def __call__(self, steps=1):
        """
        Returns the constant value for any given step

        Parameters
        ----------
            steps : int, optional
                number of steps taken (default is 1)
        """
        return self.val


class LinearSchedule:
    """
    A class used to represent a linear schedule.

    ...

    Attributes
    ----------
    start : float
        start point of the linear schedule
    end : float, optional
        end point of the linear schedule (default is None)    
    steps: int, optional
        number of steps for the linear schedule (default is None)
    inc: float
        increment value calculated from the start and end points

    Methods
    -------
    __init__(start, end=None, steps=None)
        Constructs all the necessary attributes for the linear schedule object.
    __call__(steps=1)
        Returns the current value for any given step
    """

    def __init__(self, start, end=None, steps=None):
        """
        Constructs all the necessary attributes for the linear schedule object.

        Parameters
        ----------
            start : float
                start point of the linear schedule
            end : float, optional
                end point of the linear schedule (default is None)
            steps: int, optional
                number of steps for the linear schedule (default is None)
        """
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
        """
        Returns the current value for any given step

        Parameters
        ----------
            steps : int, optional
                number of steps taken (default is 1)
        """
        val = self.current
        self.current = self.bound(self.current + self.inc * steps, self.end)
        return val