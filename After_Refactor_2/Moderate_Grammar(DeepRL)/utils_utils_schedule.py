class ConstantSchedule:
    def __init__(self, val: float):
        self.val = val

    def __call__(self, steps: int = 1) -> float:
        return self.val


class LinearSchedule:
    def __init__(self, start: float, end: Optional[float] = None, steps: Optional[float] = None):
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

    def __call__(self, steps: int = 1) -> float:
        val = self.current
        self.current = self.bound(self.current + self.inc * steps, self.end)
        return val

class Schedule:
  def __init__(self, start: float, end: Optional[float] = None, steps: Optional[float] = None) -> None:
    if end is None:
        end = start
        steps = 1
    self.current = start
    self.end = end
    self.steps = steps

  def __call__(self, steps: int = 1) -> float:
    val = self.current
    self.current = self._calculate_current(steps)
    return val

  def _calculate_current(self, steps: int) -> float:
    raise NotImplementedError

class ConstantSchedule(Schedule):
    def __init__(self, val: float) -> None:
        super().__init__(val)

    def _calculate_current(self, steps: int) -> float:
        return self.current


class LinearSchedule(Schedule):
    def __init__(self, start: float, end: Optional[float] = None, steps: Optional[float] = None) -> None:
        super().__init__(start, end, steps)
        self.inc = (end - start) / float(self.steps)
        if end > start:
            self.bound = min
        else:
            self.bound = max

    def _calculate_current(self, steps: int) -> float:
        return self.bound(self.current + self.inc * steps, self.end)

MIN_VALUE = min
MAX_VALUE = max

class Schedule:
  def __init__(self, start: float, end: Optional[float] = None, steps: Optional[float] = None) -> None:
    if end is None:
        end = start
        steps = 1
    self.current = start
    self.end = end
    self.steps = steps

  def __call__(self, steps: int = 1) -> float:
    val = self.current
    self.current = self._calculate_current(steps)
    return val

  def _calculate_current(self, steps: int) -> float:
    raise NotImplementedError

class ConstantSchedule(Schedule):
    def __init__(self, val: float) -> None:
        super().__init__(val)

    def _calculate_current(self, steps: int) -> float:
        return self.current


class LinearSchedule(Schedule):
    def __init__(self, start: float, end: Optional[float] = None, steps: Optional[float] = None) -> None:
        super().__init__(start, end, steps)
        self.inc = (end - start) / float(self.steps)
        if end > start:
            self.bound = MIN_VALUE
        else:
            self.bound = MAX_VALUE

    def _calculate_current(self, steps: int) -> float:
        return self.bound(self.current + self.inc * steps, self.end)