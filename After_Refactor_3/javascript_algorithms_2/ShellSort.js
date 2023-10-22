// To improve the maintainability of the codebase, we can refactor it by following good coding practices and applying meaningful variable names. Here is the refactored code:

/*
 * Shell Sort sorts an array based on the insertion sort algorithm
 * more information: https://en.wikipedia.org/wiki/Shellsort
 *
 */
export function shellSort(items) {
  let interval = 1;

  // Calculate the initial interval
  while (interval < items.length / 3) {
    interval = interval * 3 + 1;
  }

  // Perform shell sort with calculated intervals
  while (interval > 0) {
    for (let outerIndex = interval; outerIndex < items.length; outerIndex++) {
      const currentValue = items[outerIndex];
      let innerIndex = outerIndex;

      // Perform insertion sort within each interval
      while (innerIndex > interval - 1 && items[innerIndex - interval] >= currentValue) {
        items[innerIndex] = items[innerIndex - interval];
        innerIndex = innerIndex - interval;
      }
      items[innerIndex] = currentValue;
    }
    interval = (interval - 1) / 3;
  }
  return items;
}

