def get_factors(n):
    """
    Returns a list of prime factors for a given number n.

    Args:
    n (int): the number to be factored

    Returns:
    List[int]: a list of prime factors of n
    """

    # base case
    if n == 1:
        return []

    # loop through possible factors
    for factor in range(2, n + 1):
        if n % factor == 0:
            # factor found, recurse on the remaining quotient
            return [factor] + get_factors(n // factor)

    # no factors found, n is prime
    return [n]