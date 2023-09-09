// To improve the maintainability of the codebase, we can make the following refactors:

// 1. Separate the heapify and sinkDown methods into private methods by prefixing them with an underscore (_).
// 2. Use a comparator function instead of directly accessing the element values in the heap methods.
// 3. Add comments to explain the purpose of each method.

// Here's the refactored code:

// ```javascript
/**
 * Author: Samarth Jain
 * Max Heap implementation in Javascript
 */

class BinaryHeap {
  constructor (compareFn) {
    this.heap = []
    this.compareFn = compareFn
  }

  /**
   * Inserts a value into the heap and restores the heap property
   */
  insert (value) {
    this.heap.push(value)
    this._heapify()
  }

  /**
   * Returns the size of the heap
   */
  size () {
    return this.heap.length
  }

  /**
   * Checks if the heap is empty
   */
  empty () {
    return this.size() === 0
  }

  /**
   * Restores the heap property after insertion
   */
  _heapify () {
    let index = this.size() - 1

    while (index > 0) {
      const element = this.heap[index]
      const parentIndex = Math.floor((index - 1) / 2)
      const parent = this.heap[parentIndex]

      if (this.compareFn(parent, element) >= 0) break
      this.heap[index] = parent
      this.heap[parentIndex] = element
      index = parentIndex
    }
  }

  /**
   * Extracts the maximum element from the heap and restores the heap property
   */
  extractMax () {
    const max = this.heap[0]
    const tmp = this.heap.pop()
    if (!this.empty()) {
      this.heap[0] = tmp
      this._sinkDown(0)
    }
    return max
  }

  /**
   * Restores the heap property after extraction
   */
  _sinkDown (index) {
    const left = 2 * index + 1
    const right = 2 * index + 2
    let largest = index
    const length = this.size()

    if (left < length && this.compareFn(this.heap[left], this.heap[largest]) > 0) {
      largest = left
    }
    if (right < length && this.compareFn(this.heap[right], this.heap[largest]) > 0) {
      largest = right
    }

    if (largest !== index) {
      const tmp = this.heap[largest]
      this.heap[largest] = this.heap[index]
      this.heap[index] = tmp
      this._sinkDown(largest)
    }
  }
}

// Example

// Comparator function for the heap
// const compareFn = (a, b) => a[0] - b[0]

// const maxHeap = new BinaryHeap(compareFn)
// maxHeap.insert([4])
// maxHeap.insert([3])
// maxHeap.insert([6])
// maxHeap.insert([1])
// maxHeap.insert([8])
// maxHeap.insert([2])
// const mx = maxHeap.extractMax()

export { BinaryHeap }
// ```

// Hope this helps!

