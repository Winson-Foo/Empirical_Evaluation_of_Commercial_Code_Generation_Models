// To improve the maintainability of the codebase, we can make the following changes:

// 1. Use more descriptive variable and function names to improve code readability.
// 2. Add comments to explain the purpose of the code and improve code comprehension.
// 3. Break down the code into smaller functions to improve code modularity and reusability.
// 4. Move the main logic of calculating primes into a separate function.
// 5. Remove the unused example code.

// Here's the refactored code:

// ```javascript
/**
 * @function getSieveOfEratosthenesPrimes
 * @description Calculates prime numbers till input number n using Sieve of Eratosthenes algorithm
 * @param {Number} n - The input integer
 * @return {Number[]} List of Primes till n.
 * @see [Sieve_of_Eratosthenes](https://www.geeksforgeeks.org/sieve-of-eratosthenes/)
 */
function getSieveOfEratosthenesPrimes(n) {
  if (n <= 1) return [];

  const primes = new Array(n + 1).fill(true);
  primes[0] = primes[1] = false;

  for (let i = 2; i * i <= n; i++) {
    if (primes[i]) {
      markMultiplesAsNonPrimes(primes, i, n);
    }
  }

  return getPrimesList(primes);
}

/**
 * @function markMultiplesAsNonPrimes
 * @description Marks multiples of a prime number as non-prime
 * @param {Boolean[]} primes - Array representing prime numbers
 * @param {Number} prime - The prime number
 * @param {Number} n - The maximum number
 */
function markMultiplesAsNonPrimes(primes, prime, n) {
  for (let j = prime * prime; j <= n; j += prime) {
    primes[j] = false;
  }
}

/**
 * @function getPrimesList
 * @description Returns the list of prime numbers
 * @param {Boolean[]} primes - Array representing prime numbers
 * @return {Number[]} List of prime numbers
 */
function getPrimesList(primes) {
  const primesList = [];
  for (let i = 0; i < primes.length; i++) {
    if (primes[i]) {
      primesList.push(i);
    }
  }
  return primesList;
}

export { getSieveOfEratosthenesPrimes };
// ```

// By making these changes, the code becomes more readable, modular, and easier to maintain.

