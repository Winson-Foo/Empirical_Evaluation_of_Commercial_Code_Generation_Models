from typing import List

# This function returns a list of prime numbers up to a specified maximum value.
def find_primes(maximum: int) -> List[int]:
    # Create an empty list to store the prime numbers.
    primes = []
    # Iterate over all numbers from 2 up to the maximum value.
    for current_number in range(2, maximum + 1):
        # Check if current_number is divisible by any previous prime numbers.
        if all(current_number % prime != 0 for prime in primes):
            # If current_number is not divisible by any previous primes, add it to the list of primes.
            primes.append(current_number)
    # Return the list of prime numbers.
    return primes 