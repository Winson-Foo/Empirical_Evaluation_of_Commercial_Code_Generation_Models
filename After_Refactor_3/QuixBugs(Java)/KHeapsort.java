// To improve the maintainability of the codebase, we can make the following changes:
// 1. Add proper comments explaining the purpose and functionality of each section of code.
// 2. Use meaningful variable and method names to improve code readability.
// 3. Use standard formatting and indentation for better code organization.
// 4. Break down the logic into smaller, more understandable functions.

// Here's the refactored code:

// ```java
package java_programs;

import java.util.ArrayList;
import java.util.List;
import java.util.PriorityQueue;

public class KHeapsort {
    /**
     * Sorts the given ArrayList using the k-heapsort algorithm.
     *
     * @param arr the ArrayList to be sorted
     * @param k   the value of k for k-heapsort
     * @return the sorted ArrayList
     */
    public static List<Integer> kHeapsort(List<Integer> arr, int k) {
        PriorityQueue<Integer> heap = new PriorityQueue<>();

        // Build heap from first k elements
        for (Integer v : arr.subList(0, k)) {
            heap.add(v);
        }

        List<Integer> output = new ArrayList<>();
        for (Integer x : arr) {
            heap.add(x);
            Integer popped = heap.poll();
            output.add(popped);
        }

        // Add remaining elements from the heap
        while (!heap.isEmpty()) {
            output.add(heap.poll());
        }

        return output;
    }
}
// ```

// By following these practices, the code becomes more readable, easier to understand, and thus more maintainable.

