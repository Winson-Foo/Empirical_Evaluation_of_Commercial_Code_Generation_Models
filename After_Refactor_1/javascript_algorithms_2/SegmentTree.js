// To improve the maintainability of the codebase, here are some suggested refactorings:

// 1. Use class fields instead of assigning properties inside the constructor.

// ```javascript
// class SegmentTree {
//   size;
//   tree;

//   constructor(arr) {
//     this.size = arr.length;
//     this.tree = new Array(2 * arr.length).fill(0);

//     this.build(arr);
//   }

//   // rest of the code
// }
// ```

// 2. Use descriptive variable names and add comments to clarify the code.

// ```javascript
class SegmentTree {
  size;
  tree;

  constructor(arr) {
    // Initialize class fields
    this.size = arr.length;
    this.tree = new Array(2 * arr.length).fill(0);

    this.build(arr);
  }

  build(arr) {
    // Insert leaf nodes in the tree
    for (let i = 0; i < this.size; i++) {
      this.tree[this.size + i] = arr[i];
    }

    // Build the tree by calculating parents
    for (let i = this.size - 1; i > 0; --i) {
      this.tree[i] = this.tree[i * 2] + this.tree[i * 2 + 1];
    }
  }

  update(index, value) {
    // Only update values in the parents of the given node being changed

    // Set value at position index
    index += this.size;
    this.tree[index] = value;

    // Move upward and update parents
    for (let i = index; i > 1; i >>= 1) {
      this.tree[i >> 1] = this.tree[i] + this.tree[i ^ 1];
    }
  }

  query(left, right) {
    // Extend right boundary for convenience
    right++;

    let res = 0;

    // Loop to find the sum in the range
    for (
      left += this.size, right += this.size;
      left < right;
      left >>= 1, right >>= 1
    ) {
      // Check the left border of the query interval

      // If it's the right child of its parent, include it in the sum
      // and move to the parent of its next node
      if ((left & 1) > 0) {
        res += this.tree[left++];
      }

      // Check the right border of the query interval
      if ((right & 1) > 0) {
        res += this.tree[--right];
      }
    }

    return res;
  }
}

export { SegmentTree };
// ```

// These changes enhance the readability and maintainability of the codebase by using more descriptive variable names and adding comments to explain the logic.

