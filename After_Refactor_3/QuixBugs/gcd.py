def gcd(first_num, second_num):
    """
    Calculates the greatest common divisor of two numbers using the Euclidean algorithm.

    Args:
        first_num: The first number.
        second_num: The second number.

    Returns:
        The greatest common divisor of the two numbers.
    """
    # Base case: if the second number is 0, return the first number.
    if second_num == 0:
        return first_num
    else:
        # Recursively call gcd with the second number as the first argument
        # and the remainder of the division of the first number by the second as the second argument.
        return gcd(second_num, first_num % second_num) 