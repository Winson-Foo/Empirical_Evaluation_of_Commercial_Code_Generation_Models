package com.facebook.react.common;

/**
 * Object wrapping an auto-expanding long[]. Like an ArrayList<Long> but without the autoboxing.
 */
public class LongArray {

  private static final double GROWTH_FACTOR = 1.8;
  private static final int MIN_CAPACITY = 1;
  private static final int NO_ELEMENTS = 0;

  private long[] longArray;
  private int size;

  public static LongArray createWithInitialCapacity(int initialCapacity) {
    return new LongArray(initialCapacity);
  }

  private LongArray(int initialCapacity) {
    longArray = new long[initialCapacity];
    size = NO_ELEMENTS;
  }

  /** Adds a long value to the end of the array. */
  public void add(long value) {
    growArrayIfNeeded();
    longArray[size++] = value;
  }

  /** Returns the long value at the specified index. */
  public long get(int index) {
    if (index >= size) {
      throw new IndexOutOfBoundsException("" + index + " >= " + size);
    }
    return longArray[index];
  }

  /** Sets the long value at the specified index. */
  public void set(int index, long value) {
    if (index >= size) {
      throw new IndexOutOfBoundsException("" + index + " >= " + size);
    }
    longArray[index] = value;
  }

  /** Returns the number of elements in the array. */
  public int size() {
    return size;
  }

  /** Returns true if the array is empty. */
  public boolean isEmpty() {
    return size == NO_ELEMENTS;
  }

  /** Removes the *last* n items of the array all at once. */
  public void dropTail(int n) {
    if (n > size) {
      throw new IndexOutOfBoundsException(
          "Trying to drop " + n + " items from array of length " + size);
    }
    size -= n;
  }

  /** Grows the array if the current size is equal to the array length. */
  private void growArrayIfNeeded() {
    if (size == longArray.length) {
      int newSize = Math.max(size + MIN_CAPACITY, (int) (size * GROWTH_FACTOR));
      long[] newArray = new long[newSize];
      System.arraycopy(longArray, 0, newArray, 0, size);
      longArray = newArray;
    }
  }
}