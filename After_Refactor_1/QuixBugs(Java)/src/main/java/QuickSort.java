// To improve the maintainability of the codebase, we can make several changes:

// 1. Use more meaningful variable names: Instead of names like `arr`, `lesser`, `greater`, and `middle`, we can use names that describe the purpose of the variables, such as `inputList`, `smallerElements`, `largerElements`, and `sortedList`.

// 2. Add comments to explain the code: It's important to include comments to explain the logic of the code and make it easier for others to understand and maintain it.

// 3. Use enhanced for loop instead of subList: Instead of using `subList`, we can use an enhanced for loop to iterate over the elements of the list.

// 4. Use generics: Instead of using `ArrayList<Integer>`, we can use the more generic `List<Integer>` for method parameters and return types.

// 5. Split the sorting logic into separate methods: By splitting the sorting logic into smaller methods, we can improve readability and maintainability.

// Here's the refactored code:

package java_programs;

import java.util.ArrayList;
import java.util.List;

public class QuickSort {

    public static List<Integer> quickSort(List<Integer> inputList) {
        if (inputList.isEmpty()) {
            return new ArrayList<>();
        }

        Integer pivot = inputList.get(0);
        List<Integer> smallerElements = new ArrayList<>();
        List<Integer> largerElements = new ArrayList<>();

        for (Integer element : inputList) {
            if (element < pivot) {
                smallerElements.add(element);
            } else if (element > pivot) {
                largerElements.add(element);
            }
        }
        
        List<Integer> sortedList = new ArrayList<>();
        sortedList.addAll(quickSort(smallerElements));
        sortedList.add(pivot);
        sortedList.addAll(quickSort(largerElements));
        
        return sortedList;
    }
}