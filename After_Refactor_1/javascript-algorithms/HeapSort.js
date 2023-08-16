// To improve the maintainability of the codebase, you can consider the following changes:

// 1. Use named imports for better readability.
// 2. Separate the logic of inserting elements into the heap into a separate function.
// 3. Use a for loop instead of forEach for inserting elements into the heap.
// 4. Remove unnecessary comments and redundant code.
// 5. Rename variables and functions to improve clarity.

// Here's the refactored code:

import Sort from '../Sort';
import MinHeap from '../../../data-structures/heap/MinHeap';

export default class HeapSort extends Sort {
  sort(originalArray) {
    const sortedArray = [];
    const minHeap = new MinHeap(this.callbacks.compareCallback);

    this.insertElementsToHeap(originalArray, minHeap);

    while (!minHeap.isEmpty()) {
      const nextMinElement = minHeap.poll();
      this.callbacks.visitingCallback(nextMinElement);
      sortedArray.push(nextMinElement);
    }

    return sortedArray;
  }

  insertElementsToHeap(array, heap) {
    for (let i = 0; i < array.length; i++) {
      const element = array[i];
      this.callbacks.visitingCallback(element);
      heap.add(element);
    }
  }
}

