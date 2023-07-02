from typing import Dict, Any, NewType, Optional, Tuple, Union


class Bounds:
    def __init__(self, lower: float, upper: float):
        self.lower = lower
        self.upper = upper


BoundsInput = Union[Bounds, Tuple[float, float]]

L0 = NewType("L0", float)
L1 = NewType("L1", float)
L2 = NewType("L2", float)
Linf = NewType("Linf", float)

Preprocessing = Optional[Dict[str, Any]]