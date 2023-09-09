// To improve the maintainability of the codebase, we can make the following changes:

// 1. Add descriptive comments for better readability.
// 2. Break down the logic into smaller functions to improve modularity.
// 3. Use meaningful variable names to enhance understanding.
// 4. Remove unnecessary comments and optimize the code.

// Here is the refactored code:

// ```javascript
/**
 * @param {number} maxNumber
 * @return {number[]}
 */
export default function sieveOfEratosthenes(maxNumber) {
  const isPrime = generatePrimeArray(maxNumber);
  const primes = findPrimes(isPrime, maxNumber);
  return primes;
}

/**
 * Generates an array of booleans to track prime numbers.
 * @param {number} maxNumber
 * @returns {boolean[]}
 */
function generatePrimeArray(maxNumber) {
  const isPrime = new Array(maxNumber + 1).fill(true);
  isPrime[0] = false;
  isPrime[1] = false;
  return isPrime;
}

/**
 * Finds and returns the prime numbers up to `maxNumber`.
 * @param {boolean[]} isPrime - Array tracking prime numbers.
 * @param {number} maxNumber
 * @returns {number[]}
 */
function findPrimes(isPrime, maxNumber) {
  const primes = [];

  for (let number = 2; number <= maxNumber; number += 1) {
    if (isPrime[number]) {
      primes.push(number);
      markNonPrimes(isPrime, number, maxNumber);
    }
  }

  return primes;
}

/**
 * Marks all multiples of a prime number as non-prime.
 * @param {boolean[]} isPrime - Array tracking prime numbers.
 * @param {number} number - Prime number to start marking multiples from.
 * @param {number} maxNumber
 */
function markNonPrimes(isPrime, number, maxNumber) {
  let nextNumber = number * number;

  while (nextNumber <= maxNumber) {
    isPrime[nextNumber] = false;
    nextNumber += number;
  }
}
// ```

// By breaking down the code into smaller functions with descriptive names, we improve the maintainability of the codebase. Additionally, the code becomes more modular and easier to understand.

