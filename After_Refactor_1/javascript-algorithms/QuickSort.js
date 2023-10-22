// To improve the maintainability of this codebase, we can follow some refactoring principles:

// 1. Improve code readability by adding comments and proper indentation.
// 2. Break down the code into smaller functions to improve modularity and reusability.
// 3. Use descriptive variable names to enhance code understandability.
// 4. Move some code to helper functions to reduce complexity.

// Here's the refactored code:

// ```javascript
import Sort from "../../CONSTANT/javascript_algorithms/Sort";

export default class QuickSort extends Sort {
  /**
   * @param {*[]} originalArray
   * @return {*[]}
   */
  sort(originalArray) {
    // Clone the original array to prevent modification.
    const array = [...originalArray];

    // If the array has less than or equal to one element, it is already sorted.
    if (array.length <= 1) {
      return array;
    }

    // Split the array into left, center, and right arrays.
    const {pivotElement, centerArray, leftArray, rightArray} = this.partition(
      array
    );

    // Sort the left and right arrays.
    const leftArraySorted = this.sort(leftArray);
    const rightArraySorted = this.sort(rightArray);

    // Join the sorted left array with the center array and the sorted right array.
    return leftArraySorted.concat(centerArray, rightArraySorted);
  }

  /**
   * @param {*[]} array
   * @return {{pivotElement: *, centerArray: *[], leftArray: *[], rightArray: *[]}}
   */
  partition(array) {
    // Take the first element of the array as a pivot.
    const pivotElement = array.shift();
    const centerArray = [pivotElement];
    const leftArray = [];
    const rightArray = [];

    while (array.length) {
      const currentElement = array.shift();

      // Call visiting callback.
      this.callbacks.visitingCallback(currentElement);

      if (this.comparator.equal(currentElement, pivotElement)) {
        centerArray.push(currentElement);
      } else if (this.comparator.lessThan(currentElement, pivotElement)) {
        leftArray.push(currentElement);
      } else {
        rightArray.push(currentElement);
      }
    }

    return {
      pivotElement,
      centerArray,
      leftArray,
      rightArray,
    };
  }
}

// ```

// In the refactored code:
// - The main sorting logic is placed inside the `sort` function.
// - The partitioning logic is separated into the `partition` function for better code organization.
// - The `partition` function returns an object with the pivot element, center array, left array, and right array, making it easier to understand and debug.
// - The `sort` function uses the `partition` function to split the array before sorting.

