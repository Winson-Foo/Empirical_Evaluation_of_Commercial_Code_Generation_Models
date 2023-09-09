// To improve the maintainability of the codebase, we can do the following refactorings:

// 1. Use a more descriptive function name: 
//    - Original: `recursiveStaircaseMEM`
//    - Refactored: `getWaysToClimbStairs`

// 2. Extract the base cases into separate functions for better readability:
//    ```javascript
//    const getBaseCaseResult = (stairsNum) => {
//      if (stairsNum <= 0) {
//        return 0;
//      }
//      if (stairsNum === 1) {
//        return 1;
//      }
//      if (stairsNum === 2) {
//        return 2;
//      }
//    };
//    ```

// 3. Use a memoization object instead of an array to improve readability and performance:
//    ```javascript
//    const memo = {};
//    ```

// 4. Move the recursive closure outside of the main function to improve readability and separation of concerns:
//    ```javascript
//    const getSteps = (stairsNum) => {
//      const baseCaseResult = getBaseCaseResult(stairsNum);
//      if (baseCaseResult !== undefined) {
//        return baseCaseResult;
//      }
//      if (memo[stairsNum]) {
//        return memo[stairsNum];
//      }
//      memo[stairsNum] = getSteps(stairsNum - 1) + getSteps(stairsNum - 2);
//      return memo[stairsNum];
//    };

//    const getWaysToClimbStairs = (totalStairs) => {
//      return getSteps(totalStairs);
//    };
//    ```

// 5. Export only the `getWaysToClimbStairs` function as the default export:
//    ```javascript
//    export default getWaysToClimbStairs;
//    ```

// Here's the refactored code:

// ```javascript
const getBaseCaseResult = (stairsNum) => {
  if (stairsNum <= 0) {
    return 0;
  }
  if (stairsNum === 1) {
    return 1;
  }
  if (stairsNum === 2) {
    return 2;
  }
};

const getSteps = (stairsNum, memo) => {
  const baseCaseResult = getBaseCaseResult(stairsNum);
  if (baseCaseResult !== undefined) {
    return baseCaseResult;
  }
  if (memo[stairsNum]) {
    return memo[stairsNum];
  }
  memo[stairsNum] = getSteps(stairsNum - 1, memo) + getSteps(stairsNum - 2, memo);
  return memo[stairsNum];
};

const getWaysToClimbStairs = (totalStairs) => {
  const memo = {};
  return getSteps(totalStairs, memo);
};

export default getWaysToClimbStairs;
// ```

