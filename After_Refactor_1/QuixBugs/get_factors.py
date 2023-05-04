def factorize(number):
    """Returns a list of prime factors of the given integer."""
    if number < 2:
        return []

    for factor in range(2, int(number ** 0.5) + 1):
        if number % factor == 0:
            return [factor] + factorize(number // factor)

    return [number]