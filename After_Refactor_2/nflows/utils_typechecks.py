def is_boolean(value: Any) -> bool:
    """Check if the value is a boolean."""
    return isinstance(value, bool)


def is_integer(value: Any) -> bool:
    """Check if the value is an integer."""
    return isinstance(value, int)


def is_positive_integer(value: Any) -> bool:
    """Check if the value is a positive integer."""
    return is_integer(value) and value > 0


def is_nonnegative_integer(value: Any) -> bool:
    """Check if the value is a non-negative integer."""
    return is_integer(value) and value >= 0


def is_power_of_two(value: Any) -> bool:
    """Check if the value is a power of two."""
    if is_positive_integer(value):
        return not value & (value - 1)
    return False