// To improve the maintainability of this codebase, you can do the following:

// 1. Use descriptive variable and function names: Rename variables and functions with more meaningful names to improve code readability. This will make it easier for other developers (including future you) to understand the purpose and functionality of each element in the code.

// 2. Add comments and documentation: Add comments to clarify the logic and intent of the code. Also, provide proper documentation for the function parameters, return values, and any important notes. This will help other developers to quickly understand the codebase and its usage.

// 3. Separate concerns: Divide the code into smaller functions or modules that handle specific tasks. This will make the code more modular and easier to maintain. Each function should have a single responsibility, making it easier to test and debug.

// 4. Follow coding conventions and style guidelines: Ensure that the code follows consistent coding conventions and style guidelines. This includes indentation, spacing, naming conventions, and code organization. Consistency in the codebase will make it easier to understand and maintain.

// Here's the refactored code with improvements based on the above suggestions:

/**
 * Finds the maximum profit from buying and selling stocks.
 * DIVIDE & CONQUER APPROACH.
 *
 * @param {number[]} prices - Array of stock prices, i.e. [7, 6, 4, 3, 1]
 * @param {function(): void} visit - Visiting callback to calculate the number of iterations.
 * @returns {number} - The maximum profit.
 */
const calculateMaxProfit = (prices, visit = () => {}) => {
  /**
   * Recursive implementation of the main function. It is hidden from the users.
   *
   * @param {boolean} isBuying - Whether we are allowed to buy or sell at the current day.
   * @param {number} day - Current day of trading (current index of prices array).
   * @returns {number} - Max profit from buying/selling.
   */
  const recursiveBuyerSeller = (isBuying, day) => {
    // Registering the recursive call visit to calculate the complexity.
    visit();

    // Quitting the recursion if this is the last day of trading (prices array ended).
    if (day === prices.length) {
      return 0;
    }

    // If we're buying, we're losing money (-1). If we're selling, we're gaining money (+1).
    const operationSign = isBuying ? -1 : +1;
    return Math.max(
      // Option 1: Don't do anything.
      recursiveBuyerSeller(isBuying, day + 1),
      // Option 2: Sell or buy at the current price.
      operationSign * prices[day] + recursiveBuyerSeller(!isBuying, day + 1),
    );
  };

  const isBuying = true;
  const day = 0;

  return recursiveBuyerSeller(isBuying, day);
};

export default calculateMaxProfit;

