def generate_primes(upper_limit):
    """
    This function generates a list of prime numbers up to a given upper limit.
    """
    primes = []
    for n in range(2, upper_limit + 1):
        if all(n % p > 0 for p in primes):
            primes.append(n)
    return primes 