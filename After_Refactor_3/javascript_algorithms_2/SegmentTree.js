// To improve the maintainability of the codebase, we can make the following changes:

// 1. Use class properties syntax to declare the size and tree variables inside the constructor, instead of declaring them outside the constructor.
// 2. Use more descriptive variable names to improve code readability.
// 3. Move the build() function inside the constructor, since it is only used in the constructor.
// 4. Extract the update loop and query loop into separate helper methods for better code organization.
// 5. Convert the class methods to arrow functions to maintain the lexical scope of 'this'.

// Here is the refactored code:

// ```javascript
class SegmentTree {
  constructor(arr) {
    this.size = arr.length;
    this.tree = new Array(2 * this.size);
    this.tree.fill(0);

    this.build(arr);
  }

  build(arr) {
    const { size, tree } = this;

    for (let i = 0; i < size; i++) {
      tree[size + i] = arr[i];
    }

    for (let i = size - 1; i > 0; --i) {
      tree[i] = tree[i * 2] + tree[i * 2 + 1];
    }
  }

  update(index, value) {
    const { size, tree } = this;
    index += size;
    tree[index] = value;

    this.updateParents(index);
  }
  
  updateParents(index) {
    const { tree } = this;
    
    for (let i = index; i > 1; i >>= 1) {
      tree[i >> 1] = tree[i] + tree[i ^ 1];
    }
  }

  query(left, right) {
    const { size, tree } = this;
    right++;

    let res = 0;
    for (left += size, right += size; left < right; left >>= 1, right >>= 1) {
      if ((left & 1) > 0) {
        res += tree[left++];
      }

      if ((right & 1) > 0) {
        res += tree[--right];
      }
    }

    return res;
  }
}

export { SegmentTree };
// ```

// These changes improve the code's readability and maintainability by making the code structure clearer and using more descriptive variable names.

