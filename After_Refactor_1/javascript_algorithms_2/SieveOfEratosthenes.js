// To improve the maintainability of the codebase, we can make the following changes:

// 1. Move the prime number calculation logic into a separate function for better organization.
// 2. Add comments to explain the purpose of each section of the code.
// 3. Use meaningful variable names to improve code readability.

// Here's the refactored code:

// ```javascript
/**
 * @function calculatePrimes
 * @description Calculates prime numbers till input number n
 * @param {Number} n - The input integer
 * @return {Number[]} List of Primes till n.
 * @see [Sieve_of_Eratosthenes](https://www.geeksforgeeks.org/sieve-of-eratosthenes/)
 */
function calculatePrimes(n) {
  if (n <= 1) return [];

  // Initialize all numbers as potentially prime
  const isPrime = new Array(n + 1).fill(true);

  // Set 0 and 1 as non-prime
  isPrime[0] = isPrime[1] = false;

  // Mark non-prime numbers
  for (let i = 2; i * i <= n; i++) {
    if (isPrime[i]) {
      for (let j = i * i; j <= n; j += i) {
        isPrime[j] = false;
      }
    }
  }

  // Collect prime numbers in the result array
  const primes = [];
  for (let i = 2; i <= n; i++) {
    if (isPrime[i]) {
      primes.push(i);
    }
  }

  return primes;
}

export { calculatePrimes };
// ```

// Note: I renamed the function from `sieveOfEratosthenes` to `calculatePrimes` to better express its purpose.

