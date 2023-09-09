package java_programs;

import java.util.ArrayList;
import java.util.List;

public class MergeSort {

    public static List<Integer> merge(List<Integer> left, List<Integer> right) {
        List<Integer> result = new ArrayList<>();
        int i = 0;
        int j = 0;

        while (i < left.size() && j < right.size()) {
            if (left.get(i) <= right.get(j)) {
                result.add(left.get(i));
                i++;
            } else {
                result.add(right.get(j));
                j++;
            }
        }
        result.addAll(left.subList(i, left.size()).isEmpty() ? right.subList(j, right.size()) : left.subList(i, left.size()));
        return result;
    }

    public static List<Integer> mergeSort(List<Integer> arr) {
        if (arr.size() <= 1) {
            return arr;
        } else {
            int middle = arr.size() / 2;
            List<Integer> left = arr.subList(0, middle);
            left = mergeSort(left);
            List<Integer> right = arr.subList(middle, arr.size());
            right = mergeSort(right);

            return merge(left, right);
        }
    }
}