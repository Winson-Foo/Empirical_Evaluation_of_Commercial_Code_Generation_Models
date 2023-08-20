// To improve the maintainability of the codebase, we can make the following changes:

// 1. Use meaningful variable and function names: Instead of using generic names like `x`, `y`, and `n`, we can use more descriptive names like `smallerNumber`, `largerNumber`, and `inputNumber`.

// 2. Add comments to explain the logic: Adding comments to explain the logic of the code can make it easier for other developers (including yourself) to understand the code.

// 3. Split the logic into smaller, reusable functions: By splitting the code into smaller, reusable functions, we can improve the readability and maintainability of the code.

// 4. Use ES6 features: Utilize the ES6 features like arrow functions and destructuring to make the code more concise.

// Here's the refactored code:

// ```
/*
    Author: sandyboypraper

    Euler Totient Function (also known as phi) is the count of numbers in {1, 2, 3, ..., n} that are
    relatively prime to n, i.e., the numbers whose Greatest Common Divisor (GCD) with n is 1.
*/

// Calculates the Greatest Common Divisor (GCD) of two numbers
const calculateGCD = (smallerNumber, largerNumber) => {
  // The GCD of two numbers can be found by finding the remainder when the larger number is divided by the smaller number.
  // The GCD of x and y is equal to the GCD of y % x and x.
  return smallerNumber === 0 ? largerNumber : calculateGCD(largerNumber % smallerNumber, smallerNumber);
};

// Calculates the Euler Totient Function (phi) for a given number
const calculateEulerTotientFunction = (inputNumber) => {
  let countOfRelativelyPrimeNumbers = 1;

  // Iterate from 2 to the inputNumber and check if each number is relatively prime to the inputNumber
  for (let iterator = 2; iterator <= inputNumber; iterator++) {
    if (calculateGCD(iterator, inputNumber) === 1) {
      countOfRelativelyPrimeNumbers++;
    }
  }

  return countOfRelativelyPrimeNumbers;
};

export { calculateEulerTotientFunction };
 

