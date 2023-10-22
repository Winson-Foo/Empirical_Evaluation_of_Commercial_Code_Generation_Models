// To improve the maintainability of the codebase, we can make the following changes:

// 1. Use meaningful variable names: 
//    - Change the variable name `isnPrime` to `isNotPrime` to improve readability and make the intention of the variable more clear.
//    - Change the variable name `primes` to `primeNumbers` to make it more descriptive.

// 2. Separate the prime number calculation logic into a separate function:
//    - Create a new function called `calculatePrimes` that takes the number `n` as a parameter and returns the prime numbers.
//    - Move the logic inside the `LinearSieve` function to the new `calculatePrimes` function.

// 3. Add comments to explain the purpose and functionality of the code:
//    - Add comments to the code to explain what each section does and its time complexity.

// Here is the refactored code with the above improvements:

// ```javascript
const calculatePrimes = (n) => {
  /*
   * Calculates prime numbers till a number n
   * Time Complexity: O(n)
   * Explanation: https://cp-algorithms.com/algebra/prime-sieve-linear.html
   * :param n: Number up to which to calculate primes
   * :return: A list containing only primes
   */
  const isNotPrime = new Array(n + 1)
  isNotPrime[0] = isNotPrime[1] = true
  const primeNumbers = []

  for (let i = 2; i <= n; i++) {
    if (!isNotPrime[i]) {
      primeNumbers.push(i)
    }
    for (const prime of primeNumbers) {
      const k = i * prime
      if (k > n) break
      isNotPrime[k] = true
      if (i % prime === 0) break
    }
  }

  return primeNumbers
}

export { calculatePrimes }
// ```

// By following these improvements, the code becomes more maintainable and easier to understand. The use of descriptive variable names and comments help to clarify the purpose and functionality of the code. Additionally, separating the prime number calculation logic into a separate function improves modularity and allows for easier testing and maintenance.

