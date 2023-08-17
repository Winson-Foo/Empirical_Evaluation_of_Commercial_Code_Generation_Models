// To improve the maintainability of the codebase, you can follow these steps:

// 1. Break down the code into smaller functions or methods: By breaking down the code into smaller functions or methods, you can make it easier to understand and maintain. Each function or method should have a single responsibility and be named appropriately.

// 2. Add meaningful comments: Add comments to explain the purpose and functionality of the code. This will make it easier for other developers (including yourself) to understand the code in the future.

// 3. Use meaningful variable names: Choose variable names that accurately describe the purpose and content of the variable. This will improve the readability and understandability of the code.

// 4. Remove redundant comments: Remove comments that are stating the obvious or unnecessary. Comments should provide value by explaining complex logic or documenting important details.

// 5. Simplify complex expressions: If there are any complex expressions or conditions, try to simplify them to make the code more readable. Consider using helper functions or breaking down the expression into smaller parts.

// Here is the refactored code with the above improvements:

import Sort from '../Sort';

export default class BubbleSort extends Sort {
  sort(originalArray) {
    const array = [...originalArray];

    for (let i = 1; i < array.length; i += 1) {
      let swapped = false;

      this.visitElement(array[i]);

      for (let j = 0; j < array.length - i; j += 1) {
        this.visitElement(array[j]);

        if (this.isInWrongOrder(array[j + 1], array[j])) {
          this.swapElements(array, j, j + 1);
          swapped = true;
        }
      }

      if (!swapped) {
        return array;
      }
    }

    return array;
  }

  visitElement(element) {
    this.callbacks.visitingCallback(element);
  }

  isInWrongOrder(element1, element2) {
    return this.comparator.lessThan(element1, element2);
  }

  swapElements(array, index1, index2) {
    [array[index1], array[index2]] = [array[index2], array[index1]];
  }
}

