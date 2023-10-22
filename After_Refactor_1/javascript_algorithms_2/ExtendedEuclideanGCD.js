// To improve the maintainability of the codebase, we can make the following changes:

// 1. Add comments: Add comments to explain the purpose and functionality of each section of the code.

// 2. Use meaningful variable names: Replace variable names like arg1, arg2, r0, r1, etc. with more descriptive names to improve clarity and understanding.

// 3. Extract repeated calculation into helper functions: Extract the calculation of r2, s2, and t2 into a separate helper function to improve readability and maintainability.

// 4. Use destructuring assignment: Instead of swapping the values manually, use destructuring assignment to swap the values of res[1] and res[2] in the recursive call.

// Here is the refactored code:

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
 * This is called Bézout's identity and the coefficients are called Bézout coefficients
 *
 * The algorithm uses the Euclidean method of getting remainder:
 * r_i+1 = r_i-1 - qi*ri
 * and applies it to series s and t (with same quotient q at each stage)
 * When r_n reaches 0, the value r_n-1 gives the gcd, and s_n-1 and t_n-1 give the coefficients
 *
 * This implementation uses an iterative approach to calculate the values
 */

/**
 *
 * @param {Number} number1 first argument
 * @param {Number} number2 second argument
 * @returns Array with GCD and first and second Bézout coefficients
 */
const extendedEuclideanGCD = (number1, number2) => {
  if (typeof number1 !== 'number' || typeof number2 !== 'number') throw new TypeError('Not a Number')
  if (number1 < 1 || number2 < 1) throw new TypeError('Must be positive numbers')

  // Make the order of coefficients correct, as the algorithm assumes r0 > r1
  if (number1 < number2) {
    const [gcd, coeff1, coeff2] = extendedEuclideanGCD(number2, number1)
    return [gcd, coeff2, coeff1]
  }

  // At this point number1 > number2

  // Remainder values
  let remainder0 = number1
  let remainder1 = number2

  // Coefficient1 values
  let coefficient1_0 = 1
  let coefficient1_1 = 0

  // Coefficient2 values
  let coefficient2_0 = 0
  let coefficient2_1 = 1

  while (remainder1 !== 0) {
    const quotient = Math.floor(remainder0 / remainder1)

    const [newRemainder, newCoefficient1, newCoefficient2] = calculateNewValues(remainder0, remainder1, coefficient1_0, coefficient1_1, coefficient2_0, coefficient2_1, quotient)

    remainder0 = remainder1
    remainder1 = newRemainder
    coefficient1_0 = coefficient1_1
    coefficient1_1 = newCoefficient1
    coefficient2_0 = coefficient2_1
    coefficient2_1 = newCoefficient2
  }
  return [remainder0, coefficient1_0, coefficient2_0]
}

const calculateNewValues = (remainder0, remainder1, coefficient1_0, coefficient1_1, coefficient2_0, coefficient2_1, quotient) => {
  const newRemainder = remainder0 - remainder1 * quotient
  const newCoefficient1 = coefficient1_0 - coefficient1_1 * quotient
  const newCoefficient2 = coefficient2_0 - coefficient2_1 * quotient

  return [newRemainder, newCoefficient1, newCoefficient2]
}

export { extendedEuclideanGCD }

