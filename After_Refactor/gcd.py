def greatest_common_divisor(a, b):
    """
    Returns the greatest common divisor of the two given integers.

    Args:
        a (int): the first integer
        b (int): the second integer

    Returns:
        int: the greatest common divisor of the two integers
    """
    if b == 0:
        return a
    else:
        return greatest_common_divisor(b, a % b)