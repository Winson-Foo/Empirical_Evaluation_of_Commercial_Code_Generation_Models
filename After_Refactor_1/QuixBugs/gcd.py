def greatest_common_divisor(number1, number2):
    """
    Returns the greatest common divisor (GCD) of two numbers using the Euclidean algorithm.
    """
    # Ensure that number1 is greater than or equal to number2
    if number2 > number1:
        number1, number2 = number2, number1
    # Base case: if number2 is 0, return number1
    if number2 == 0:
        return number1
    # Recursive case: calculate the GCD of number2 and the remainder of number1 divided by number2
    else:
        remainder = number1 % number2
        return greatest_common_divisor(number2, remainder)