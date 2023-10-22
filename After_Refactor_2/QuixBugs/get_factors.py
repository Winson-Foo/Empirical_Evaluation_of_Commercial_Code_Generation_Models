def get_factors(n):
    factors = []
    factor = 2
    while factor <= n:
        if n % factor == 0:
            factors.append(factor)
            n = n // factor
        else:
            factor += 1
    return factors