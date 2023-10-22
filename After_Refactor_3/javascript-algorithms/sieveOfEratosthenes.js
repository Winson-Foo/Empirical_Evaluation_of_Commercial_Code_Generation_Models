// To improve the maintainability of this codebase, we can make the following changes:

// 1. Use meaningful variable names: Instead of using generic names like `isPrime` and `primes`, we can use more descriptive names that convey their purpose.

// 2. Break down complex expressions: The condition `isPrime[number] === true` can be simplified to `isPrime[number]`.

// 3. Add comments to explain the code logic and any potential issues.

// Here's the refactored code with these changes:

// ```javascript
/**
 * Generate an array of prime numbers up to a given maxNumber
 * @param {number} maxNumber - The maximum number up to which to generate prime numbers
 * @return {number[]} - An array of prime numbers
 */
export default function sieveOfEratosthenes(maxNumber) {
  // Create an array to track whether a number is prime or not
  const isPrime = new Array(maxNumber + 1).fill(true);
  isPrime[0] = false;
  isPrime[1] = false;

  const primes = [];

  // Iterate through all numbers starting from 2
  for (let number = 2; number <= maxNumber; number += 1) {
    // If the number is prime
    if (isPrime[number]) {
      primes.push(number);

      // Optimisation: Mark multiples of `number` as non-prime starting from `number * number`
      let nextNumber = number * number;

      // Mark all multiples of `number` as non-prime
      while (nextNumber <= maxNumber) {
        isPrime[nextNumber] = false;
        nextNumber += number;
      }
    }
  }

  return primes;
}
// ```

// By applying these changes, the code becomes easier to read, understand, and maintain.

