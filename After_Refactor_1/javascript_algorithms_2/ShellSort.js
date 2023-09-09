// One way to improve the maintainability of the codebase is to break the code into smaller, more focused functions. This will make the code easier to understand and modify in the future. Here is an example of refactored code that utilizes helper functions:

// ```javascript
export function shellSort(items) {
  let interval = calculateInterval(items.length)

  while (interval > 0) {
    for (let outer = interval; outer < items.length; outer++) {
      const value = items[outer]
      let inner = outer

      while (inner > interval - 1 && items[inner - interval] >= value) {
        items[inner] = items[inner - interval]
        inner = inner - interval
      }
      items[inner] = value
    }
    interval = calculateNextInterval(interval)
  }
  return items
}

function calculateInterval(length) {
  let interval = 1
  while (interval < length / 3) {
    interval = interval * 3 + 1
  }
  return interval
}

function calculateNextInterval(interval) {
  return (interval - 1) / 3
}
// ```

// In this refactored code, `calculateInterval()` and `calculateNextInterval()` are helper functions that encapsulate the logic for calculating the interval. This makes the main `shellSort()` function cleaner and easier to read.

