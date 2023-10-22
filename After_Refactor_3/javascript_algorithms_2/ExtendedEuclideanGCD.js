// To improve the maintainability of the codebase, we can make the following refactors:

// 1. Add proper comments to explain the purpose and functionality of each section of code.
// 2. Split the code into small, concise functions with descriptive names.
// 3. Use destructuring assignment to improve readability.
// 4. Use more meaningful variable names.
// 5. Add type annotations to function parameters and return values.
// 6. Handle edge cases and input validations explicitly.

// Here's the refactored code:

// ```javascript
/**
 * Problem statement and explanation: https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm
 *
 * This algorithm plays an important role for modular arithmetic and cryptography algorithms.
 *
 * Basic explanation:
 * The Extended Euclidean algorithm is a modification of the standard Euclidean GCD algorithm.
 * It allows to calculate coefficients x and y for the equation:
 * ax + by = gcd(a,b)
 *
 * This is called B�zout's identity, and the coefficients are called B�zout coefficients.
 *
 * The algorithm uses the Euclidean method of getting remainder:
 * r_i+1 = r_i-1 - qi*ri,
 * and applies it to series s and t with the same quotient q at each stage.
 * When r_n reaches 0, the value r_n-1 gives the gcd, and s_n-1 and t_n-1 give the coefficients.
 *
 * This implementation uses an iterative approach to calculate the values.
 */

/**
 * Calculates the extended Euclidean GCD and B�zout coefficients for two numbers.
 *
 * @param {number} a - The first argument.
 * @param {number} b - The second argument.
 * @returns {number[]} - Array with GCD and first and second B�zout coefficients.
 * @throws {TypeError} - If either argument is not a positive number.
 */
const extendedEuclideanGCD = (a, b) => {
  if (typeof a !== 'number' || typeof b !== 'number') {
    throw new TypeError('Both arguments must be numbers.')
  }

  if (a < 1 || b < 1) {
    throw new TypeError('Both arguments must be positive numbers.')
  }

  // Make the order of coefficients correct, as the algorithm assumes r0 > r1
  if (a < b) {
    const [gcd, coef1, coef2] = extendedEuclideanGCD(b, a)
    return [gcd, coef2, coef1]
  }

  // At this point, a > b

  // Initialize values
  let r0 = a
  let r1 = b
  let s0 = 1
  let s1 = 0
  let t0 = 0
  let t1 = 1

  // Iterate until r1 becomes 0
  while (r1 !== 0) {
    const quotient = Math.floor(r0 / r1)
    const remainder = r0 - r1 * quotient
    const coef1 = s0 - s1 * quotient
    const coef2 = t0 - t1 * quotient

    // Update values for next iteration
    r0 = r1
    r1 = remainder
    s0 = s1
    s1 = coef1
    t0 = t1
    t1 = coef2
  }

  // Return GCD and B�zout coefficients
  return [r0, s0, t0]
}

export { extendedEuclideanGCD }
// ```

// These improvements make the code easier to understand, maintain, and test.

