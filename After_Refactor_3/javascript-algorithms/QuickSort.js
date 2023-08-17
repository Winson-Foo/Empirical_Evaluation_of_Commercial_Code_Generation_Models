// To improve the maintainability of the codebase, we can make the following changes:

// 1. Improve code readability by using meaningful variable names and comments.
// 2. Move the logic for splitting the array and sorting into separate functions.
// 3. Use recursion to simplify the sorting process.

// Here is the refactored code:

// ```javascript
import Sort from '../Sort';

export default class QuickSort extends Sort {
  /**
   * @param {*[]} originalArray
   * @return {*[]}
   */
  sort(originalArray) {
    // Clone original array to prevent it from modification.
    const array = [...originalArray];

    // If array has less than or equal to one elements then it is already sorted.
    if (array.length <= 1) {
      return array;
    }

    // Split the array into left, center, and right arrays.
    const { leftArray, centerArray, rightArray } = this.splitArray(array);

    // Sort the left and right arrays recursively.
    const leftArraySorted = this.sort(leftArray);
    const rightArraySorted = this.sort(rightArray);

    // Combine the sorted left array, center array, and sorted right array.
    return [...leftArraySorted, ...centerArray, ...rightArraySorted];
  }

  /**
   * Split the array into left, center, and right arrays based on the pivot element.
   *
   * @param {*[]} array - The array to be split.
   * @return {{ leftArray: *[], centerArray: *[], rightArray: *[] }}
   */
  splitArray(array) {
    const leftArray = [];
    const centerArray = [];
    const rightArray = [];

    const pivotElement = array[0];  // Take the first element of the array as the pivot.

    for (let i = 0; i < array.length; i++) {
      const currentElement = array[i];

      // Call the visiting callback.
      this.callbacks.visitingCallback(currentElement);

      if (this.comparator.equal(currentElement, pivotElement)) {
        centerArray.push(currentElement);
      } else if (this.comparator.lessThan(currentElement, pivotElement)) {
        leftArray.push(currentElement);
      } else {
        rightArray.push(currentElement);
      }
    }

    return { leftArray, centerArray, rightArray };
  }
}
// ```

// By separating the logic into separate functions and using meaningful variable names, the code is easier to understand and maintain. Recursion is used to simplify the sorting process, making it easier to follow the flow of the algorithm.

