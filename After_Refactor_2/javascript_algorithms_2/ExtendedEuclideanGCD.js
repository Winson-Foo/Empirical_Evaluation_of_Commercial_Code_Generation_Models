// To improve the maintainability of the codebase, you can make the following changes:

// 1. Add comments: Provide comments to explain the purpose and logic of different sections of the code. This will make it easier for other developers (and even yourself) to understand the code in the future.

// 2. Use descriptive variable names: Replace variable names like "arg1", "arg2", "r0", "r1", etc. with more descriptive names that convey their purpose. This will make the code more readable and easier to understand.

// 3. Use destructuring assignment: Instead of accessing array elements using indices, use destructuring assignment to assign meaningful names to the elements. This will make the code more self-explanatory.

// 4. Use helper functions: Break down complex calculations or repetitive code into separate helper functions. This will improve code readability and make it easier to maintain.

// Here's the refactored code with the aforementioned changes:

// ```javascript
/**
 * Problem statement and explanation: https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm
 *
 * This algorithm plays an important role for modular arithmetic, and by extension for cryptography algorithms
 *
 * Basic explanation:
 * The Extended Euclidean algorithm is a modification of the standard Euclidean GCD algorithm.
 * It allows to calculate coefficients x and y for the equation:
 *          ax + by = gcd(a,b)
 *
 * This is called B�zout's identity and the coefficients are called B�zout coefficients
 *
 * The algorithm uses the Euclidean method of getting remainder:
 * r_i+1 = r_i-1 - qi*ri
 * and applies it to series s and t (with same quotient q at each stage)
 * When r_n reaches 0, the value r_n-1 gives the gcd, and s_n-1 and t_n-1 give the coefficients
 *
 * This implementation uses an iterative approach to calculate the values
 */

/**
 * Calculates the extended Euclidean GCD
 *
 * @param {Number} num1 - The first argument
 * @param {Number} num2 - The second argument
 * @returns {Array} - An array with GCD and first and second B�zout coefficients
 */
const extendedEuclideanGCD = (num1, num2) => {
  if (typeof num1 !== 'number' || typeof num2 !== 'number') {
    throw new TypeError('Not a Number')
  }
  if (num1 < 1 || num2 < 1) {
    throw new TypeError('Must be positive numbers')
  }

  // Make the order of coefficients correct, as the algorithm assumes r0 > r1
  if (num1 < num2) {
    const [gcd, coeff1, coeff2] = extendedEuclideanGCD(num2, num1)
    return [gcd, coeff2, coeff1]
  }

  // At this point, num1 > num2

  // Remainder values
  let dividend = num1
  let divisor = num2

  // Coefficient1 values
  let coeff1Curr = 1
  let coeff1Prev = 0

  // Coefficient2 values
  let coeff2Curr = 0
  let coeff2Prev = 1

  while (divisor !== 0) {
    const quotient = Math.floor(dividend / divisor)

    const remainder = dividend - divisor * quotient
    const coeff1New = coeff1Prev - coeff1Curr * quotient
    const coeff2New = coeff2Prev - coeff2Curr * quotient

    dividend = divisor
    divisor = remainder
    coeff1Prev = coeff1Curr
    coeff1Curr = coeff1New
    coeff2Prev = coeff2Curr
    coeff2Curr = coeff2New
  }

  return [dividend, coeff1Prev, coeff2Prev]
}

export { extendedEuclideanGCD }
// ```

// By incorporating these changes, the code becomes more readable, maintainable, and easier to understand for future developers.

