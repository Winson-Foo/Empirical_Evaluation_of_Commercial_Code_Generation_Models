"""Functions that check types."""

from typing import Union

def is_bool(value: Union[bool, int]) -> bool:
    return isinstance(value, bool)

def is_int(value: Union[bool, int]) -> bool:
    return isinstance(value, int)

def is_positive_int(value: Union[bool, int]) -> bool:
    return isinstance(value, int) and value > 0

def is_nonnegative_int(value: Union[bool, int]) -> bool:
    return isinstance(value, int) and value >= 0

def is_power_of_two(value: Union[bool, int]) -> bool:
    if is_positive_int(value):
        return not value & (value - 1)
    else:
        return False