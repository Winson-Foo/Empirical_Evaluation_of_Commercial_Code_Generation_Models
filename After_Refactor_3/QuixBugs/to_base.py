def convert_decimal_to_base(decimal_number, base):
    """
    Converts a decimal number to its equivalent in the given base number system.

    :param decimal_number: The decimal number to convert.
    :type decimal_number: int
    :param base: The base of the number system to convert to.
    :type base: int
    :return: The equivalent number in the specified base number system.
    :rtype: str
    """
    digits = get_digits_in_base_system(base)
    result = ''
    while decimal_number > 0:
        remainder = decimal_number % base
        decimal_number //= base
        result = digits[remainder] + result
    return result

def get_digits_in_base_system(base):
    """
    Generates the digits for the given base number system.

    :param base: The base of the number system.
    :type base: int
    :return: A string containing the digits for the specified base number system.
    :rtype: str
    """
    if base <= 1:
        raise ValueError("Base must be greater than 1.")
    if base > 36:
        raise ValueError("Base cannot be greater than 36.")
    digits = string.digits + string.ascii_uppercase
    return digits[:base] 