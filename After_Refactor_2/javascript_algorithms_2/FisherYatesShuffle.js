// To improve the maintainability of this codebase, you can make the following changes:

// 1. Use descriptive variable names: Instead of using "maxLength" and "idx", use more meaningful names such as "arrayLength" and "randomIndex". This makes the code easier to understand and maintain.

// 2. Extract the swapping logic into a separate function: Create a helper function called "swap" that takes the array and two indices as parameters. This function will swap the elements at the given indices in the array.

// 3. Use array destructuring assignment: Instead of using a temporary variable to swap the elements, you can use array destructuring assignment. This simplifies the code and makes it more readable.

// Here's the refactored code with these improvements:

export const shuffle = (array) => {
  let arrayLength = array.length;

  const swap = (arr, i, j) => {
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }

  while (arrayLength) {
    let randomIndex = Math.floor(Math.random() * arrayLength--);
    swap(array, arrayLength, randomIndex);
  }

  return array;
}

