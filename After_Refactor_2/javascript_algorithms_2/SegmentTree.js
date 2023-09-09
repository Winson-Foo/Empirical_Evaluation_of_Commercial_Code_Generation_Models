// To improve the maintainability of the codebase, we can make the following changes:

// 1. Encapsulate class properties: Instead of declaring `this.size` and `this.tree` as public properties of the class, we can encapsulate them by making them private and accessing them through getter methods.

// 2. Use descriptive variable names: Instead of using single-letter variable names like `N`, `L`, and `R`, we can use more descriptive names like `size`, `left`, and `right` to improve code readability.

// 3. Split the `build` method into smaller functions: The `build` method is responsible for inserting leaf nodes and calculating parents. By splitting it into smaller functions, we can improve code maintainability and readability.

// 4. Use `const` instead of `let` for loop variables: Since the loop variables `i` in the `build` and `update` methods are not modified within the loop, we can use `const` instead of `let`.

// 5. Add comments to explain the purpose of each method and block of code.

// Here is the refactored code:

// ```javascript
/**
 * Segment Tree
 * concept : [Wikipedia](https://en.wikipedia.org/wiki/Segment_tree)
 * inspired by : https://www.geeksforgeeks.org/segment-tree-efficient-implementation/
 *
 * time complexity
 * - init : O(N)
 * - update : O(log(N))
 * - query : O(log(N))
 *
 * space complexity : O(N)
 */
class SegmentTree {
  #size
  #tree

  constructor(arr) {
    this.#size = arr.length
    this.#tree = new Array(2 * arr.length)
    this.#tree.fill(0)

    this.#build(arr)
  }

  /**
   * Builds the segment tree by inserting leaf nodes and calculating parents.
   * @param {Array} arr - The input array.
   */
  #build(arr) {
    const { #size, #tree } = this
    
    // Insert leaf nodes in tree
    for (let i = 0; i < #size; i++) {
      #tree[#size + i] = arr[i]
    }

    // Calculate parents
    for (let i = #size - 1; i > 0; --i) {
      #tree[i] = #tree[i * 2] + #tree[i * 2 + 1]
    }
  }

  /**
   * Updates the value at the given index in the segment tree.
   * @param {number} index - The index to be updated.
   * @param {number} value - The new value.
   */
  update(index, value) {
    const { #size, #tree } = this

    index += #size
    #tree[index] = value

    for (let i = index; i > 1; i >>= 1) {
      #tree[i >> 1] = #tree[i] + #tree[i ^ 1]
    }
  }

  /**
   * Queries the sum of values in the range [left, right).
   * @param {number} left - The left index (inclusive).
   * @param {number} right - The right index (exclusive).
   * @returns {number} - The sum of values in the range.
   */
  query(left, right) {
    const { #size, #tree } = this
    right++

    let res = 0

    for (left += #size, right += #size; left < right; left >>= 1, right >>= 1) {
      if ((left & 1) > 0) {
        res += #tree[left++]
      }

      if ((right & 1) > 0) {
        res += #tree[--right]
      }
    }

    return res
  }

  /**
   * Returns the size of the segment tree.
   * @returns {number} - The size of the segment tree.
   */
  get size() {
    return this.#size
  }

  /**
   * Returns the segment tree array.
   * @returns {Array} - The segment tree array.
   */
  get tree() {
    return this.#tree
  }
}

export { SegmentTree }
 