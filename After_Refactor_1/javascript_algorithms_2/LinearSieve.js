// To improve the maintainability of the codebase, here are some suggested changes:

// 1. Add comments: Add comments to explain the purpose of the code, the algorithm being used, and the logic behind each step. This will make it easier for others (and yourself in the future) to understand the code.

// 2. Use more descriptive variable names: Replace variables like "n" and "p" with more descriptive names that reflect their purpose in the code. This will make it easier to understand what each variable represents.

// 3. Extract helper functions: Extract the inner loop that checks for prime numbers into a separate helper function. This will make the code more modular and easier to understand.

// Here is the refactored code with these improvements:

// ```javascript
const LinearSieve = (n) => {
  /*
   * Calculates prime numbers till a number n
   * Time Complexity: O(n)
   * Explanation: https://cp-algorithms.com/algebra/prime-sieve-linear.html
   * :param n: Number up to which to calculate primes
   * :return: A list containing only primes
   */
  
  // Initialize the array to check if a number is prime
  const isNotPrime = new Array(n + 1)
  isNotPrime[0] = isNotPrime[1] = true
  
  const primes = []
  
  // Loop through numbers from 2 to n
  for (let num = 2; num <= n; num++) {
    if (!isNotPrime[num]) {
      primes.push(num)
      markMultiples(num, n, isNotPrime)
    }
  }
  
  return primes
}

// Helper function to mark multiples of prime numbers as not prime
function markMultiples(prime, n, isNotPrime) {
  for (let i = prime * prime; i <= n; i += prime) {
    isNotPrime[i] = true
  }
}

export { LinearSieve }
// ```

// With these changes, the code should be more readable, self-explanatory, and easier to maintain and understand.

