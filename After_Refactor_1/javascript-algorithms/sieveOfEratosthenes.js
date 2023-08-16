// To improve the maintainability of the codebase, you can refactor it as follows:

// ```
/**
 * @param {number} maxNumber
 * @return {number[]}
 */
export default function sieveOfEratosthenes(maxNumber) {
  const isPrime = new Array(maxNumber + 1).fill(true);
  isPrime[0] = false;
  isPrime[1] = false;

  const primes = [];

  for (let number = 2; number <= maxNumber; number += 1) {
    if (isPrime[number]) {
      primes.push(number);
      markMultiplesAsNonPrime(number, maxNumber, isPrime);
    }
  }

  return primes;
}

function markMultiplesAsNonPrime(number, maxNumber, isPrime) {
  let nextNumber = number * number;

  while (nextNumber <= maxNumber) {
    isPrime[nextNumber] = false;
    nextNumber += number;
  }
}
// ```

// Refactoring the code involves extracting the logic for marking multiples of a prime number as non-prime into a separate `markMultiplesAsNonPrime` function. This improves maintainability by separating concerns and making the code more modular and readable.

// Additionally, I replaced the comparison `isPrime[number] === true` with `isPrime[number]` in the if statement, as the value is already a boolean.

// By following these refactoring changes, the code is more maintainable and easier to understand and modify in the future.

