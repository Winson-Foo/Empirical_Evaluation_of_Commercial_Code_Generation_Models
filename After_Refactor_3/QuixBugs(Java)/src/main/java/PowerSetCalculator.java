// To improve the maintainability of the codebase, we can make several changes:

// 1. Use meaningful names for variables and methods.
// 2. Add proper comments to explain the functionality of the code.
// 3. Encapsulate the code into a class instead of using a standalone method.
// 4. Use generics to make the code more type-safe.
// 5. Reduce the complexity of the code by removing unnecessary duplication and excessive use of ArrayList.

// Here is the refactored code with these improvements:

// ```java
package java_programs;
import java.util.ArrayList;
import java.util.List;

/**
 * This class provides a method to calculate the power set of a given list.
 */
public class PowerSetCalculator<T> {
  
    /**
     * Returns the power set of a given list.
     * 
     * @param list The input list
     * @return The power set of the input list
     */
    public List<List<T>> calculatePowerSet(List<T> list) {
        List<List<T>> powerSet = new ArrayList<>();
        calculatePowerSetHelper(list, 0, new ArrayList<>(), powerSet);
        return powerSet;
    }

    private void calculatePowerSetHelper(List<T> list, int index, List<T> subset, List<List<T>> powerSet) {
        powerSet.add(subset);

        for (int i = index; i < list.size(); i++) {
            List<T> newSubset = new ArrayList<>(subset);
            newSubset.add(list.get(i));
            calculatePowerSetHelper(list, i + 1, newSubset, powerSet);
        }
    }
}
// ```

// Now, you can use the `PowerSetCalculator` class to calculate the power set of any list. For example:

// ```java
// List<Integer> inputList = Arrays.asList(1, 2, 3);
// PowerSetCalculator<Integer> calculator = new PowerSetCalculator<>();
// List<List<Integer>> powerSet = calculator.calculatePowerSet(inputList);

// // Print the power set
// for (List<Integer> subset : powerSet) {
//     System.out.println(subset);
// }
// ```

