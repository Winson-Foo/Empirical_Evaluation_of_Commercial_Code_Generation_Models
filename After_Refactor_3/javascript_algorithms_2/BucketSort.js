// To improve the maintainability of this codebase, you can do the following refactoring:

// 1. Use meaningful variable names to improve code readability.
// 2. Break down the bucketSort function into smaller, more focused functions.
// 3. Replace hard-coded numbers with constant variables.
// 4. Remove unnecessary comments.
// 5. Use arrow functions and array methods for cleaner code.

// Here's the refactored code:

// ```javascript
/**
 * BucketSort implementation.
 *
 * Wikipedia says: Bucket sort, or bin sort, is a sorting algorithm that works by distributing the elements of an array
 * into a number of buckets. Each bucket is then sorted individually, either using a different sorting algorithm, or by
 * recursively applying the bucket sorting algorithm. It is a distribution sort, and is a cousin of radix sort in the
 * most to least significant digit flavour. Bucket sort is a generalization of pigeonhole sort. Bucket sort can be
 * implemented with comparisons and therefore can also be considered a comparison sort algorithm. The computational
 * complexity estimates involve the number of buckets.
 *
 * @see https://en.wikipedia.org/wiki/Bucket_sort#:~:text=Bucket%20sort%2C%20or%20bin%20sort,applying%20the%20bucket%20sorting%20algorithm.&text=Sort%20each%20non%2Dempty%20bucket.
 *
 * Time Complexity of Solution:
 * Best Case O(n); Average Case O(n); Worst Case O(n)
 *
 * @param {number[]} list The array of numbers to be sorted.
 * @param {number} size The size of the buckets used. If not provided, size will be 5.
 * @return {number[]} An array of numbers sorted in increasing order.
 */

export function bucketSort(list, size = 5) {
  if (list.length === 0) {
    return list;
  }

  const min = findMin(list);
  const max = findMax(list);

  const count = Math.floor((max - min) / size) + 1;
  const buckets = createBuckets(count);

  fillBuckets(list, buckets, min, size);

  const sorted = mergeBuckets(buckets);

  return sorted;
}

function findMin(list) {
  return Math.min(...list);
}

function findMax(list) {
  return Math.max(...list);
}

function createBuckets(count) {
  return Array.from({ length: count }, () => []);
}

function fillBuckets(list, buckets, min, size) {
  for (let i = 0; i < list.length; i++) {
    const key = Math.floor((list[i] - min) / size);
    buckets[key].push(list[i]);
  }
}

function mergeBuckets(buckets) {
  return buckets
    .map((bucket) => bucket.sort((a, b) => a - b))
    .reduce((sorted, bucket) => sorted.concat(bucket), []);
}
 

