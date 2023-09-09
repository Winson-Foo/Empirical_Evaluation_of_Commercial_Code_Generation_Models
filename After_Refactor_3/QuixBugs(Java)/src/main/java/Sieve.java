package java_programs;

import java.util.ArrayList;
import java.util.List;

/**
 * The Sieve class implements the Sieve of Eratosthenes algorithm to find all prime numbers up to a given maximum value.
 */
public class Sieve {

    /**
     * Checks if all elements in the list are true.
     *
     * @param list the list to check
     * @return true if all elements are true, false otherwise
     */
    public static boolean allTrue(List<Boolean> list) {
        for (boolean value : list) {
            if (!value) {
                return false;
            }
        }
        return true;
    }

    /**
     * Checks if any element in the list is true.
     *
     * @param list the list to check
     * @return true if any element is true, false otherwise
     */
    public static boolean anyTrue(List<Boolean> list) {
        for (boolean value : list) {
            if (value) {
                return true;
            }
        }
        return false;
    }

    /**
     * Generates a list comprehension based on the given number and prime numbers list.
     *
     * @param number the number to generate list comprehension for
     * @param primes the list of prime numbers
     * @return the list comprehension
     */
    public static List<Boolean> generateListComprehension(int number, List<Integer> primes) {
        List<Boolean> listComprehension = new ArrayList<Boolean>();
        for (Integer prime : primes) {
            listComprehension.add(number % prime > 0);
        }
        return listComprehension;
    }

    /**
     * Finds all prime numbers up to the given maximum value using the Sieve of Eratosthenes algorithm.
     *
     * @param max the maximum value to search for prime numbers
     * @return the list of prime numbers
     */
    public static List<Integer> sieve(int max) {
        List<Integer> primes = new ArrayList<Integer>();
        for (int number = 2; number <= max; number++) {
            if (anyTrue(generateListComprehension(number, primes))) {
                primes.add(number);
            }
        }
        return primes;
    }
}