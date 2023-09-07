package java_programs;

import java.util.ArrayList;

/**
 * A class that implements the Sieve of Eratosthenes algorithm to find prime numbers.
 * The code has been refactored for improved maintainability.
 */
public class SieveOfEratosthenes {

    /**
     * Checks if all values in the given ArrayList are true.
     *
     * @param arr The ArrayList to check.
     * @return True if all values are true, false otherwise.
     */
    public static boolean allTrue(ArrayList<Boolean> arr) {
        for (boolean value : arr) {
            if (!value) {
                return false;
            }
        }
        return true;
    }

    /**
     * Checks if any value in the given ArrayList is true.
     *
     * @param arr The ArrayList to check.
     * @return True if any value is true, false otherwise.
     */
    public static boolean anyTrue(ArrayList<Boolean> arr) {
        for (boolean value : arr) {
            if (value) {
                return true;
            }
        }
        return false;
    }

    /**
     * Builds an ArrayList of boolean values using list comprehension.
     * Each value represents whether the corresponding prime number is a factor of the given number.
     *
     * @param n      The number to check for factors.
     * @param primes The ArrayList of prime numbers.
     * @return The ArrayList of boolean values.
     */
    public static ArrayList<Boolean> buildFactorList(int n, ArrayList<Integer> primes) {
        ArrayList<Boolean> factorList = new ArrayList<>();
        for (Integer p : primes) {
            factorList.add(n % p > 0);
        }
        return factorList;
    }

    /**
     * Finds all prime numbers up to the given maximum value using the Sieve of Eratosthenes algorithm.
     *
     * @param max The maximum value to check for prime numbers.
     * @return The ArrayList of prime numbers.
     */
    public static ArrayList<Integer> findPrimes(int max) {
        ArrayList<Integer> primes = new ArrayList<>();
        for (int n = 2; n <= max; n++) {
            if (anyTrue(buildFactorList(n, primes))) {
                primes.add(n);
            }
        }
        return primes;
    }
}