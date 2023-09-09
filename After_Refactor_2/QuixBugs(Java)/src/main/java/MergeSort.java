package java_programs;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * Class representing a Merge Sort algorithm implementation.
 */
public class MergeSort {
    /**
     * Merge two sorted lists into a single sorted list.
     *
     * @param left  The left sorted list
     * @param right The right sorted list
     * @return The merged sorted list
     */
    public static List<Integer> merge(List<Integer> left, List<Integer> right) {
        List<Integer> mergedList = new ArrayList<>();

        Iterator<Integer> leftIterator = left.iterator();
        Iterator<Integer> rightIterator = right.iterator();

        Integer leftElement = getNextElement(leftIterator);
        Integer rightElement = getNextElement(rightIterator);

        while (leftElement != null && rightElement != null) {
            if (leftElement <= rightElement) {
                mergedList.add(leftElement);
                leftElement = getNextElement(leftIterator);
            } else {
                mergedList.add(rightElement);
                rightElement = getNextElement(rightIterator);
            }
        }

        appendRemainingElements(mergedList, leftElement, leftIterator);
        appendRemainingElements(mergedList, rightElement, rightIterator);

        return mergedList;
    }

    /**
     * Recursive function to perform Merge Sort on a list of integers.
     *
     * @param elements The list of integers to be sorted
     * @return The sorted list of integers
     */
    public static List<Integer> mergeSort(List<Integer> elements) {
        if (elements.size() <= 1) {
            return elements;
        }

        int middle = elements.size() / 2;

        List<Integer> left = mergeSort(elements.subList(0, middle));
        List<Integer> right = mergeSort(elements.subList(middle, elements.size()));

        return merge(left, right);
    }

    /**
     * Helper function to get the next element from an iterator or null if there are no more elements.
     *
     * @param iterator The iterator to get the next element from
     * @return The next element or null if there are no more elements
     */
    private static Integer getNextElement(Iterator<Integer> iterator) {
        if (iterator.hasNext()) {
            return iterator.next();
        } else {
            return null;
        }
    }

    /**
     * Helper function to append the remaining elements from an iterator to a list.
     *
     * @param list     The list to append the elements to
     * @param element  The current element
     * @param iterator The iterator containing the remaining elements
     */
    private static void appendRemainingElements(List<Integer> list, Integer element, Iterator<Integer> iterator) {
        while (element != null) {
            list.add(element);
            element = getNextElement(iterator);
        }
    }
}