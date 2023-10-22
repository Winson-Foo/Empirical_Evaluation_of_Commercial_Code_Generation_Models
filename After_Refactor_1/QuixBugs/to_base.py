import string

def convert_to_base(number: int, base: int) -> str:
    """
    Converts a positive integer to a string in the given base.

    Args:
        number: The positive integer to convert.
        base: The base to convert the integer to.

    Returns:
        A string representation of the number in the given base.
    """
    if number < 0 or base < 2:
        raise ValueError('Number must be positive and base must be at least 2.')

    symbols = string.digits + string.ascii_uppercase
    result = ''

    while number > 0:
        remainder = number % base
        number //= base
        result = f'{symbols[remainder]}{result}'

    return result or '0'  # Handle the special case of number = 0.