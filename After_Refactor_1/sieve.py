def sieve(max_number):
    primes = []
    for number in range(2, max_number + 1):
        if not any(number % prime == 0 for prime in primes):
            primes.append(number)
    return primes
