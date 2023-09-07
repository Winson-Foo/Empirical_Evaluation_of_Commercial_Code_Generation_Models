// Here is the refactored code with improved maintainability:

// ```java
package java_programs;
import java.util.ArrayList;
import java.util.List;

public class QuickSort {
    public static ArrayList<Integer> quicksort(ArrayList<Integer> arr) {
        if (arr.isEmpty()) {
            return new ArrayList<Integer>();
        }

        Integer pivot = arr.get(0);
        List<Integer> lesser = new ArrayList<Integer>();
        List<Integer> greater = new ArrayList<Integer>();

        for (Integer x : arr.subList(1, arr.size())) {
            if (x < pivot) {
                lesser.add(x);
            } else if (x > pivot) {
                greater.add(x);
            }
        }
        
        List<Integer> middle = new ArrayList<Integer>();
        middle.add(pivot);
        lesser = quicksort(new ArrayList<>(lesser));
        greater = quicksort(new ArrayList<>(greater));
        middle.addAll(greater);
        lesser.addAll(middle);
        return new ArrayList<>(lesser);
    }
}
// ```

// Changes made:
// - Renamed the class to QuickSort to follow Java naming convention.
// - Changed the variable types from specific ArrayList to more generic types (List).
// - Updated the ArrayList creation to use the List interface.
// - Used explicit types when creating new ArrayList instances for recursive calls.
// - Added additional line breaks and indentation for better code readability.

