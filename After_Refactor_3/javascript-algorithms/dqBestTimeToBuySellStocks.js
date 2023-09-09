// To improve the maintainability of the codebase, we can make the following changes:

// 1. Add meaningful comments: Add comments to explain the purpose and logic of each section of the code.

// 2. Use descriptive variable names: Change variable names that are not descriptive to more meaningful ones, so it is easier to understand the code.

// 3. Break down complex functions: Divide the code into smaller, more focused functions to improve readability and maintainability.

// 4. Use constants instead of hardcoded values: Replace hardcoded values with constants to make the code more flexible and easier to modify.

// 5. Remove unnecessary parameters: Remove unnecessary parameters from the function definition to simplify the logic.

// Here is the refactored code:

/**
 * Finds the maximum profit from buying and selling stocks.
 * DIVIDE & CONQUER APPROACH.
 *
 * @param {number[]} prices - Array of stock prices, i.e. [7, 6, 4, 3, 1]
 * @param {function(): void} visit - Visiting callback to calculate the number of iterations.
 * @return {number} - The maximum profit
 */
const dqBestTimeToBuySellStocks = (prices, visit = () => {}) => {
  /**
   * Recursive implementation of the main function.
   * It calculates the max profit from buying and selling stocks recursively.
   *
   * @param {boolean} buy - Whether we're allowing to sell or to buy now
   * @param {number} day - Current day of trading (current index of the prices array)
   * @returns {number} - Max profit from buying/selling
   */
  const recursiveBuyerSeller = (buy, day) => {
    // Registering the recursive call visit to calculate the complexity.
    visit();

    // Quitting the recursion if this is the last day of trading (prices array ended).
    if (day === prices.length) {
      return 0;
    }

    // If we're buying, we're losing money (-1), if we're selling, we're getting money (+1).
    const operationSign = buy ? -1 : +1;
    
    const doNothingProfit = recursiveBuyerSeller(buy, day + 1);
    const buySellProfit = operationSign * prices[day] + recursiveBuyerSeller(!buy, day + 1);

    return Math.max(doNothingProfit, buySellProfit);
  };

  const initialBuy = true;
  const initialDay = 0;

  return recursiveBuyerSeller(initialBuy, initialDay);
};

export default dqBestTimeToBuySellStocks;

