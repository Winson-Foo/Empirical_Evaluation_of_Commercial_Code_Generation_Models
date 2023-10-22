// To improve the maintainability of this codebase, you can follow these refactoring steps:

// 1. Add proper comments: Add comments to explain the purpose and logic of each part of the code.

// ```javascript
// const LinearSieve = (n) => {
//   /*
//    * Calculates prime numbers till a number n
//    * Time Complexity: O(n)
//    * Explanation: https://cp-algorithms.com/algebra/prime-sieve-linear.html
//    * :param n: Number up to which to calculate primes
//    * :return: A list containing only primes
//    */
//   // ...
// ```

// 2. Use meaningful variable names: Choose clear and meaningful names for variables and functions to improve readability.

// ```javascript
//   const isNotPrime = new Array(n + 1);
//   isNotPrime[0] = isNotPrime[1] = true;
//   const primes = [];
// ```

// 3. Extract nested loops into separate functions: Extract the nested loop into a separate function for better readability and maintainability.

// ```javascript
//   for (let i = 2; i <= n; i++) {
//     if (!isNotPrime[i]) primes.push(i);
//     markNonPrimes(i);
//   }
// ```

// 4. Split complex operations into smaller functions: Split the complex operation of marking non-prime numbers into a separate function.

// ```javascript
//   function markNonPrimes(i) {
//     for (const p of primes) {
//       const k = i * p;
//       if (k > n) break;
//       isNotPrime[k] = true;
//       if (i % p === 0) break;
//     }
//   }
// ```

// 5. Return the result: Finally, return the primes array.

// ```javascript
//   return primes;
// };

// export { LinearSieve };
// ```

// Here is the refactored code:

// ```javascript
const LinearSieve = (n) => {
  /*
   * Calculates prime numbers till a number n
   * Time Complexity: O(n)
   * Explanation: https://cp-algorithms.com/algebra/prime-sieve-linear.html
   * :param n: Number up to which to calculate primes
   * :return: A list containing only primes
   */

  const isNotPrime = new Array(n + 1);
  isNotPrime[0] = isNotPrime[1] = true;
  const primes = [];

  for (let i = 2; i <= n; i++) {
    if (!isNotPrime[i]) primes.push(i);
    markNonPrimes(i);
  }

  function markNonPrimes(i) {
    for (const p of primes) {
      const k = i * p;
      if (k > n) break;
      isNotPrime[k] = true;
      if (i % p === 0) break;
    }
  }

  return primes;
};

export { LinearSieve };
// ```

// By following these steps, the code becomes more readable and maintainable.

