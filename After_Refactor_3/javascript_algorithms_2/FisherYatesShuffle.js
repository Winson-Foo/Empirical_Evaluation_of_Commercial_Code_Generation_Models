// To improve the maintainability of the codebase, we can make the following changes:

// 1. Rename the variables to have more descriptive names.
// 2. Use destructuring assignment to swap elements in the array.
// 3. Move the "while" loop to a "for" loop for clearer iteration.
// 4. Add a try-catch block to handle potential errors.

// Here is the refactored code:

export const shuffle = (array) => {
  try {
    let arrayLength = array.length;

    for (let i = arrayLength - 1; i > 0; i--) {
      let randomIndex = Math.floor(Math.random() * (i + 1));

      [array[i], array[randomIndex]] = [array[randomIndex], array[i]];
    }

    return array;
  } catch (error) {
    // Handle the error here or rethrow it
    console.error("An error occurred while shuffling the array:", error);
    throw error;
  }
}

