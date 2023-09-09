// To improve the maintainability of the codebase, you can make the following changes:

// 1. Add descriptive comments: Add comments to explain the purpose and functionality of the code, as well as any important details.

// 2. Use meaningful variable names: Update variable names to be more descriptive and reflect their purpose in the code.

// 3. Implement error handling: Instead of throwing a `TypeError`, consider using a `try-catch` block to handle errors and provide more informative error messages.

// 4. Extract helper functions: Extract any repetitive or complex calculations into helper functions to improve readability and maintainability.

// Here is the refactored code with these improvements:

// Calculates the mean squared error between predicted and expected arrays
const meanSquaredError = (predicted, expected) => {
  // Check if the arguments are arrays
  if (!Array.isArray(predicted) || !Array.isArray(expected)) {
    throw new TypeError('Both arguments must be arrays');
  }

  // Check if the arrays have equal length
  if (predicted.length !== expected.length) {
    throw new TypeError('Both arrays must have equal length');
  }

  let errorSum = 0;

  // Calculate the squared error for each element
  for (let i = 0; i < expected.length; i++) {
    const squaredError = calculateSquaredError(predicted[i], expected[i]);
    errorSum += squaredError;
  }

  // Calculate the mean squared error
  const meanError = errorSum / expected.length;

  return meanError;
};

// Helper function to calculate the squared error
const calculateSquaredError = (predictedValue, expectedValue) => {
  return (expectedValue - predictedValue) ** 2;
};

export { meanSquaredError };

