def is_prime(n, primes):
    """
    Checks if a number is prime or not.
    
    Args:
    - n (int): the number to be checked
    - primes (list): a list of prime numbers
    
    Returns:
    - True if the number is prime
    - False otherwise
    """
    for p in primes:
        if n % p == 0:
            return False
    return True

def sieve(max_number):
    """
    Returns a list of prime numbers up to a given maximum number.
    
    Args:
    - max_number (int): the maximum number to check
    
    Returns:
    - a list of prime numbers
    """
    primes = []
    for n in range(2, max_number + 1):
        if is_prime(n, primes):
            primes.append(n)
    return primes